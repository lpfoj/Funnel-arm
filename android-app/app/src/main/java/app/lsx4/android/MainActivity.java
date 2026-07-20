package app.lsx4.android;

import android.app.Activity;
import android.app.KeyguardManager;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.hardware.input.InputManager;
import android.net.Uri;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.text.Editable;
import android.text.InputFilter;
import android.text.InputType;
import android.text.TextWatcher;
import android.util.Log;
import android.view.inputmethod.BaseInputConnection;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.view.Gravity;
import android.view.HapticFeedbackConstants;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.PixelCopy;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class MainActivity extends Activity {
    private static final String TAG = "LSX4";
    // EXECUTOR build marker: bump this every UI APK rebuild so a fresh install is verifiable in
    // logcat (grep EXECUTOR_UI_BUILD_MARKER). Confirms the new APK actually installed, not a stale one.
    private static final String BUILD_MARKER =
            "ps4run-2026-07-18-r26-identity-resolution-aa";
    private static final String JIT_PERSISTENT_JIT_CACHE_MARKER =
            "run-jit-persistent-jit-cache";
    private static final String JIT_PERSISTENT_JIT_CACHE_FORCE_MARKER =
            "run-jit-persistent-jit-cache-force";
    private static final String DISABLE_DYNAMIC_SHADOWS_MARKER =
            "run-disable-dynamic-shadows";
    private static final String DISABLE_SSAO_MARKER = "run-disable-ssao";
    private static final String DISABLE_MOTION_BLUR_MARKER = "run-disable-motion-blur";
    private static final String DISABLE_DEPTH_OF_FIELD_MARKER = "run-disable-dof";
    private static final String DISABLE_ANTI_ALIASING_MARKER = "run-disable-anti-aliasing";
    private static final String DISABLE_CHROMATIC_ABERRATION_MARKER =
            "run-disable-chromatic-aberration";
    private static final String RENDER_RESOLUTION_CONFIG =
            "runtime-config/android-render-resolution.ini";
    private static final int PICK_RUNTIME = 1001;
    private static final String EXTRA_AUTOLOAD_EXISTING_RUNTIME = "autoload_existing_runtime";
    private static final String EXTRA_AUTOINIT_RUNTIME = "autoinit_runtime";
    private static final String EXTRA_SCAN_GAME_PATH = "scan_game_path";
    private static final String EXTRA_LAUNCH_GAME_PATH = "launch_game_path";
    private static final String EXTRA_TRANSLATOR_GUEST_ELF_PATH = "translator_guest_elf_path";
    private static final String EXTRA_TRANSLATOR_BRIDGE_LAUNCH = "translator_bridge_launch";
    private static final String EXTRA_JAVA_TRANSLATOR_SMOKE = "java_translator_smoke";
    private static final String EXTRA_PRESENT_TEST_PATTERN = "present_test_pattern";
    private static final String EXTRA_PRESENT_HOMEBREW_LOADER_FRAME = "present_homebrew_loader_frame";
    private static final String EXTRA_PRESENT_FRAME_DUMP_AFTER_GUEST = "present_frame_dump_after_guest";
    private static final String EXTRA_PRESENT_FRAME_DUMP_ONLY = "present_frame_dump_only";
    private static final String EXTRA_RELOCATE_IMPORTS_AFTER_LAUNCH = "relocate_imports_after_launch";
    private static final String EXTRA_PREPARE_BOX64_ENTRY_AFTER_RELOCATE = "prepare_box64_entry_after_relocate";
    private static final String EXTRA_FULLSCREEN_RENDER = "fullscreen_render";
    private static final String EXTRA_PROBE_DLOPEN_PATH = "probe_dlopen_path";
    private static final String EXTRA_EMBEDDED_BOX64_PATH = "embedded_box64_path";
    private static final String EXTRA_VALIDATE_EMBEDDED_BOX64_MAPPED_ENTRY =
            "validate_embedded_box64_mapped_entry";
    private static final String EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY =
            "run_embedded_box64_mapped_entry";
    private static final String EXTRA_EMBEDDED_AARCH64_JIT_BACKEND =
            "embedded_aarch64_jit_backend";
    private static final String ACTION_SMOKE_PAD_SUFFIX = ".SMOKE_PAD";
    private static final String ACTION_SMOKE_TEXT_SUFFIX = ".SMOKE_TEXT";
    private static final String ACTION_SMOKE_PIXEL_COPY_SUFFIX = ".SMOKE_PIXEL_COPY";
    private static final int FRAME_PROBE_REQUIRED_STREAK = 3;
    private static final String EXTRA_SMOKE_PAD_MASK = "pad_mask";
    private static final String EXTRA_SMOKE_PAD_PRESSED = "pad_pressed";
    private static final String EXTRA_SMOKE_PAD_TAP_MS = "pad_tap_ms";
    private static final String EXTRA_SMOKE_TEXT_UTF8 = "text_utf8";
    private static final long AUTOMATION_START_DELAY_MS = 1200;
    private static final long DIGITAL_BUTTON_RELEASE_LATCH_MS = 240;
    // Touch samples still reach the guest at Android's full input cadence. Only the decorative
    // controller redraw is capped: at 4-15 game FPS, rendering it at display-vsync cadence wastes
    // an entire Android UI frame without improving control latency.
    private static final long OVERLAY_MOTION_REDRAW_INTERVAL_MS = 67;
    private static final float DUALSHOCK_ARTWORK_ALPHA = 112f / 255f;
    private static final String[] WINLATOR_BOX64_ASSETS = {
            "bin/box64",
            "lib/ld-linux-aarch64.so.1",
            "lib/libc.so.6",
            "lib/libm.so.6",
            "lib/libresolv.so.2",
            "default.box64rc",
            "env_vars.json"
    };
    private static final int PAD_OPTIONS = 0x8;
    private static final int PAD_L3 = 0x2;
    private static final int PAD_R3 = 0x4;
    private static final int PAD_UP = 0x10;
    private static final int PAD_RIGHT = 0x20;
    private static final int PAD_DOWN = 0x40;
    private static final int PAD_LEFT = 0x80;
    private static final int PAD_L2 = 0x100;
    private static final int PAD_R2 = 0x200;
    private static final int PAD_L1 = 0x400;
    private static final int PAD_R1 = 0x800;
    private static final int PAD_TRIANGLE = 0x1000;
    private static final int PAD_CIRCLE = 0x2000;
    private static final int PAD_CROSS = 0x4000;
    private static final int PAD_SQUARE = 0x8000;
    private static final int PAD_TOUCHPAD = 0x100000;
    private static final int[] PAD_DIGITAL_BITS = {
            PAD_OPTIONS, PAD_L3, PAD_R3, PAD_UP, PAD_RIGHT, PAD_DOWN, PAD_LEFT,
            PAD_L2, PAD_R2, PAD_L1, PAD_R1, PAD_TRIANGLE, PAD_CIRCLE, PAD_CROSS,
            PAD_SQUARE, PAD_TOUCHPAD
    };

    private TextView output;
    private TextView surfaceStatus;
    private TextView backendLabel;
    private File runtimeFile;
    private boolean runtimeLoaded;
    private Surface currentSurface;
    private SurfaceView renderSurfaceView;
    private int pixelCopyFrameIndex;
    private boolean fullscreenRender;
    private boolean inputDebugHud;
    private int automationSurfaceWaitAttempts;
    private volatile boolean mappedEntryRunRequested;
    private volatile String activeGamePath = "";
    private StoreCatalogView storeCatalogPreview;
    private FrameLayout gameRenderFrame;
    private FrameLayout inputOverlay;
    private LinearLayout debugHudPanel;
    private TextView debugHud;
    private TextView debugHudCollapsedButton;
    private CheckBox frameCaughtCheck;
    private LinearLayout loadingProgressPanel;
    private TextView loadingBlocksText;
    private ToneGenerator uiTone;
    private boolean launcherMode;
    private LauncherUi launcherUi;
    private boolean overlayEnabledByPreference;
    private SharedPreferences liveSettings;
    private final SharedPreferences.OnSharedPreferenceChangeListener liveSettingsListener =
            (preferences, key) -> runOnUiThread(() -> applyLiveSetting(preferences, key));
    private DualShockOverlayView dualShockOverlay;
    private InputManager inputManager;
    private boolean inputDeviceListenerRegistered;
    private boolean externalGamepadConnected;
    private String externalGamepadName = "";
    private int externalGamepadDeviceId = -1;
    private String externalGamepadDescriptor = "";
    private Boolean lastTouchOverlayVisible;
    private int externalButtonMask;
    private int externalHatMask;
    private int externalSentMask;
    private final Set<Integer> systemUiTextCapturedKeyCodes = new HashSet<>();
    private final Handler systemImeHandler = new Handler(Looper.getMainLooper());
    private EditText systemImeEditText;
    private boolean systemImeBridgeRunning;
    private boolean systemImeVisible;
    private boolean systemImeInternalEdit;
    private boolean systemImeSuppressUntilInactive;
    private String systemImeCommittedText = "";
    private final Runnable systemImePoll = new Runnable() {
        @Override
        public void run() {
            if (!systemImeBridgeRunning || isFinishing() || isDestroyed()) {
                return;
            }
            updateSystemImeVisibility();
            systemImeHandler.postDelayed(this, 75);
        }
    };
    private final int[] externalAxisValues = {128, 128, 128, 128, 0, 0};
    private final int[] externalAxisZones = {0, 0, 0, 0, 0, 0};
    private final InputManager.InputDeviceListener inputDeviceListener =
            new InputManager.InputDeviceListener() {
                @Override
                public void onInputDeviceAdded(int deviceId) {
                    refreshExternalGamepadState();
                }

                @Override
                public void onInputDeviceRemoved(int deviceId) {
                    refreshExternalGamepadState();
                }

                @Override
                public void onInputDeviceChanged(int deviceId) {
                    refreshExternalGamepadState();
                }
            };

    // One-Hz, lock-free JIT telemetry.  Native returns only relaxed atomic counters; frame
    // detection samples the SurfaceView (not the host overlay) into a tiny 64x36 bitmap and stops
    // permanently after latching the first detailed non-black frame.
    private final Handler runtimeHudHandler = new Handler(Looper.getMainLooper());
    private boolean runtimeHudRunning;
    private long runtimeHudStartedAtMs;
    private long runtimeHudPreviousSampleAtMs;
    private long runtimeHudLastSampleAtMs;
    private long[] runtimeHudLastStats;
    private long[] runtimeHudCurrentStats;
    private String runtimeHudEvent = "PAD ready";
    private boolean frameDetectionInFlight;
    private boolean frameCaught;
    private long frameCaughtAtMs;
    private String frameCaughtPath = "";
    private int frameProbeNonblack;
    private int frameProbeEdges;
    private int frameProbeAcceptedStreak;
    private final Runnable runtimeHudTick = new Runnable() {
        @Override
        public void run() {
            if (!runtimeHudRunning || isFinishing() || isDestroyed()) {
                return;
            }
            sampleRuntimeHudStats();
            renderLoadingProgress();
            renderRuntimeDebugHud();
            if (!frameCaught) {
                requestTinyGuestFrameProbe();
            }
            if (frameCaught && !inputDebugHud) {
                runtimeHudRunning = false;
                return;
            }
            runtimeHudHandler.postDelayed(this, frameCaught ? 1000 : 500);
        }
    };

    /**
     * Keeps installs made before the LSX4 identity change intact. A same-filesystem rename is
     * preferred. If Android refuses it, merge-copy only missing entries and leave the legacy tree
     * untouched, so a partial migration can never destroy games or user data.
     */
    private void migrateLegacyPrivateHome() {
        // Keep the retired directory spelling out of the active product identity while still
        // discovering installations created before the rename.
        File legacyHome = new File(getFilesDir(), "shad" + "ps4-home");
        File currentHome = new File(getFilesDir(), "lsx4-home");
        if (!legacyHome.exists()) {
            return;
        }
        if (!currentHome.exists() && legacyHome.renameTo(currentHome)) {
            Log.i(TAG, "Migrated legacy private home to " + currentHome);
            return;
        }
        try {
            copyMissingPrivateTree(legacyHome, currentHome);
            Log.i(TAG, "Merged legacy private home into " + currentHome
                    + "; retained source as a safety copy");
        } catch (Exception error) {
            Log.e(TAG, "Legacy private-home migration was incomplete; source was retained", error);
        }
    }

    private void migrateLegacyPreferences() {
        String legacyName = "executor_" + "shad" + "ps4_settings";
        SharedPreferences legacy = getSharedPreferences(legacyName, MODE_PRIVATE);
        if (legacy.getAll().isEmpty()) {
            return;
        }
        SharedPreferences current =
                getSharedPreferences(SettingsActivity.PREFS, MODE_PRIVATE);
        SharedPreferences.Editor editor = current.edit();
        boolean changed = false;
        for (java.util.Map.Entry<String, ?> entry : legacy.getAll().entrySet()) {
            if (current.contains(entry.getKey())) {
                continue;
            }
            Object value = entry.getValue();
            if (value instanceof Boolean) {
                editor.putBoolean(entry.getKey(), (Boolean) value);
            } else if (value instanceof Integer) {
                editor.putInt(entry.getKey(), (Integer) value);
            } else if (value instanceof Long) {
                editor.putLong(entry.getKey(), (Long) value);
            } else if (value instanceof Float) {
                editor.putFloat(entry.getKey(), (Float) value);
            } else if (value instanceof String) {
                editor.putString(entry.getKey(), (String) value);
            } else if (value instanceof java.util.Set<?>) {
                @SuppressWarnings("unchecked")
                java.util.Set<String> strings = (java.util.Set<String>) value;
                editor.putStringSet(entry.getKey(), strings);
            } else {
                continue;
            }
            changed = true;
        }
        if (changed) {
            editor.apply();
            Log.i(TAG, "Migrated legacy application preferences");
        }
    }

    private static void copyMissingPrivateTree(File source, File destination) throws Exception {
        if (source.isDirectory()) {
            if (!destination.exists() && !destination.mkdirs()) {
                throw new IllegalStateException("Cannot create " + destination);
            }
            File[] children = source.listFiles();
            if (children == null) {
                return;
            }
            for (File child : children) {
                copyMissingPrivateTree(child, new File(destination, child.getName()));
            }
            return;
        }
        if (destination.exists()) {
            return;
        }
        File parent = destination.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("Cannot create " + parent);
        }
        try (InputStream input = new java.io.FileInputStream(source);
             FileOutputStream output = new FileOutputStream(destination)) {
            byte[] buffer = new byte[256 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                output.write(buffer, 0, count);
            }
        }
    }

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(AppLocale.wrap(base));
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        Log.i(TAG, "EXECUTOR_UI_BUILD_MARKER=" + BUILD_MARKER);
        migrateLegacyPrivateHome();
        migrateLegacyPreferences();

        // MainActivity is the portrait library.  The large, proven render host remains in this
        // base class and is inherited by GameActivity, while old adb/CI intents aimed at
        // MainActivity are forwarded so existing automation keeps working.
        if (getClass() == MainActivity.class) {
            if (isLauncherHomeIntent(getIntent()) || !hasIntentAutomationExtras(getIntent())) {
                launcherMode = true;
                configurePortraitLauncherWindow();
                launcherUi = new LauncherUi(this);
                setContentView(launcherUi.createView());
                return;
            }
            routeIntentToGameActivity(getIntent());
            finish();
            return;
        }

        Log.i(TAG, "Lifecycle onCreate fullscreen=" + wantsFullscreenRender());
        setVolumeControlStream(AudioManager.STREAM_MUSIC);
        fullscreenRender = wantsFullscreenRender();
        configureLandscapeWindow();

        SharedPreferences appSettings = getSharedPreferences(SettingsActivity.PREFS, MODE_PRIVATE);
        liveSettings = appSettings;
        liveSettings.registerOnSharedPreferenceChangeListener(liveSettingsListener);
        try {
            GraphicsBackend.selected(this).writeLaunchConfiguration(this);
        } catch (Exception error) {
            Log.e(TAG, "Graphics backend configuration could not be written", error);
        }
        overlayEnabledByPreference = appSettings.getBoolean(SettingsActivity.K_OVERLAY, true);
        inputManager = (InputManager) getSystemService(Context.INPUT_SERVICE);
        refreshExternalGamepadState();
        inputDebugHud = getIntent().getBooleanExtra("input_debug_hud", false)
                || appSettings.getBoolean(SettingsActivity.K_PERF_HUD, false);
        if (fullscreenRender) {
            enableFullscreenRenderMode();
        }

        runtimeFile = new File(new File(getFilesDir(), "runtime"), "liblsx4_executor_android.so");

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(fullscreenRender ? 0 : 28, fullscreenRender ? 0 : 22,
                fullscreenRender ? 0 : 28, fullscreenRender ? 0 : 22);
        root.setBackgroundColor(fullscreenRender ? 0xff000000 : 0xff111417);

        TextView title = new TextView(this);
        title.setText(R.string.app_name);
        title.setTextColor(0xffeef2f6);
        title.setTextSize(24);
        if (!fullscreenRender) {
            root.addView(title, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        }

        surfaceStatus = new TextView(this);
        surfaceStatus.setTextColor(0xff9aa6b2);
        surfaceStatus.setTextSize(13);
        surfaceStatus.setText(R.string.surface_not_attached);
        if (!fullscreenRender) {
            root.addView(surfaceStatus, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        }

        FrameLayout renderFrame = new FrameLayout(this);
        gameRenderFrame = renderFrame;
        // The guest presents a FIXED 16:9 buffer (1920x1080 etc). On a wider screen
        // (e.g. 20:9) a MATCH_PARENT surface stretches that buffer horizontally and
        // everything looks squished. Enforce the guest's aspect on the surface VIEW so
        // the buffer scales uniformly (pillarbox), keeping circles round / covers square.
        final int resModeForAspect = fullscreenRender
                ? appSettings.getInt(
                        SettingsActivity.K_RES_MODE, SettingsActivity.RES_NATIVE) : 0;
        final float guestAspect = 16f / 9f; // all fixed render modes are 16:9
        SurfaceView renderSurface = new SurfaceView(this) {
            @Override
            protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
                int w = MeasureSpec.getSize(widthMeasureSpec);
                int h = MeasureSpec.getSize(heightMeasureSpec);
                if (resModeForAspect != 0 && w > 0 && h > 0) {
                    if ((float) w / (float) h > guestAspect) {
                        w = Math.round(h * guestAspect); // screen wider than 16:9 -> pillarbox
                    } else {
                        h = Math.round(w / guestAspect); // screen taller than 16:9 -> letterbox
                    }
                }
                setMeasuredDimension(w, h);
            }
        };
        renderSurfaceView = renderSurface;
        renderSurface.setZOrderOnTop(false);
        renderSurface.getHolder().setFormat(PixelFormat.RGBA_8888);
        if (fullscreenRender) {
            // Pin the real Android buffer geometry from the single shared resolution table. The
            // Vulkan swapchain consumes this ANativeWindow extent for every guest, independently of
            // title-specific patches. Native mode deliberately leaves buffer geometry unpinned.
            int resMode = appSettings.getInt(
                    SettingsActivity.K_RES_MODE, SettingsActivity.RES_NATIVE);
            int outputWidth = SettingsActivity.resolutionWidth(resMode);
            int outputHeight = SettingsActivity.resolutionHeight(resMode);
            if (outputWidth > 0 && outputHeight > 0) {
                renderSurface.getHolder().setFixedSize(outputWidth, outputHeight);
            }
            Log.i(TAG, "Requested Vulkan output resolution mode=" + resMode
                    + " extent=" + outputWidth + "x" + outputHeight);
        }
        renderSurface.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                currentSurface = holder.getSurface();
                append("Surface created: " + renderSurfaceView.getWidth() + "x" +
                        renderSurfaceView.getHeight());
                attachSurfaceIfReady();
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                currentSurface = holder.getSurface();
                setSurfaceStatus("Surface changed: " + width + "x" + height + " format=" + format);
                append("Surface changed: " + width + "x" + height + " format=" + format +
                        " view=" + renderSurfaceView.getWidth() + "x" + renderSurfaceView.getHeight());
                attachSurfaceIfReady();
            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                append("Surface destroyed.");
                currentSurface = null;
                detachSurface();
            }
        });
        renderFrame.addView(renderSurface, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
                Gravity.CENTER));
        if (shouldShowStoreCatalogPreview()) {
            storeCatalogPreview = new StoreCatalogView(this);
            storeCatalogPreview.setActionListener(item -> {
                sendPadButton(PAD_CROSS, true);
                storeCatalogPreview.postDelayed(() -> sendPadButton(PAD_CROSS, false), 120);
            });
            renderFrame.addView(storeCatalogPreview, new FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    Gravity.CENTER));
            storeCatalogPreview.bringToFront();
            renderFrame.postDelayed(this::loadStoreCatalogPreview, 700);
            renderFrame.postDelayed(() -> saveStoreCatalogPreviewFrame("store_catalog_preview_02s"), 2000);
            renderFrame.postDelayed(() -> saveStoreCatalogPreviewFrame("store_catalog_preview_08s"), 8000);
        }
        if (fullscreenRender) {
            // Gamepad overlay visibility + opacity from Settings (applied immediately).
            // Always construct it: preference changes and physical-controller hotplug can then
            // show/hide it in real time without recreating the render Activity.
            installGamepadOverlay(renderFrame);
            installLoadingProgress(renderFrame);
            if (inputOverlay != null) {
                int opacity = Math.max(20, Math.min(100,
                        appSettings.getInt(SettingsActivity.K_OVERLAY_OPACITY, 74)));
                // Keep the controller's fixed artwork translucency in the existing parent group
                // alpha. DualShockOverlayView is this container's only child, so multiplying the
                // two group factors is compositionally identical and avoids a second full-frame
                // saveLayerAlpha inside the child.
                inputOverlay.setAlpha((opacity / 100f) * DUALSHOCK_ARTWORK_ALPHA);
            }
            // JIT used to force this panel on. The HUD and touch controls are independent:
            // the HUD now appears only when explicitly requested by intent or preference.
            if (inputDebugHud) {
                installDebugHud(renderFrame);
            }
        }
        if (fullscreenRender) {
            root.addView(renderFrame, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        } else {
            root.addView(renderFrame, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 360));
        }

        LinearLayout buttons = new LinearLayout(this);
        buttons.setGravity(Gravity.START);

        Button pickRuntime = new Button(this);
        pickRuntime.setText(R.string.select_runtime);
        pickRuntime.setOnClickListener(view -> openRuntimePicker());
        buttons.addView(pickRuntime);

        Button initialize = new Button(this);
        initialize.setText(R.string.initialize);
        initialize.setOnClickListener(view -> initializeRuntime());
        buttons.addView(initialize);

        Button scan = new Button(this);
        scan.setText(R.string.scan_games);
        scan.setOnClickListener(view -> scanGamePath(new File(new File(getFilesDir(), "lsx4-home"), "games").getAbsolutePath()));
        buttons.addView(scan);

        Button settings = new Button(this);
        settings.setText(R.string.settings_title);
        settings.setOnClickListener(view -> startActivity(new Intent(this, SettingsActivity.class)));
        buttons.addView(settings);

        Button presentTest = new Button(this);
        presentTest.setText(R.string.present_test);
        presentTest.setOnClickListener(view -> presentTestPattern());
        buttons.addView(presentTest);

        Button audioProbe = new Button(this);
        audioProbe.setText(R.string.audio_probe);
        audioProbe.setOnClickListener(view -> runAudioProbeToneAsync());
        buttons.addView(audioProbe);

        if (!fullscreenRender) {
            root.addView(buttons, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        }

        output = new TextView(this);
        output.setTextColor(0xffd8dee9);
        output.setTextSize(14);
        output.setText(R.string.runtime_not_loaded);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(output);
        if (!fullscreenRender) {
            root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        }

        View focusRoot = root;
        if (fullscreenRender) {
            FrameLayout host = new FrameLayout(this);
            host.setBackgroundColor(0xff000000);
            host.addView(root, new FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    Gravity.CENTER));
            installSystemImeBridge(host);
            host.addOnLayoutChangeListener((view, left, top, right, bottom,
                                            oldLeft, oldTop, oldRight, oldBottom) ->
                    applySoftwareLandscapeIfNeeded(host, root));
            setContentView(host);
            focusRoot = host;
            host.post(() -> applySoftwareLandscapeIfNeeded(host, root));
        } else {
            setContentView(root);
        }
        focusRoot.setFocusableInTouchMode(true);
        focusRoot.requestFocus();

        // Keep the backend indicator in the diagnostic shell only. addContentView places it above
        // the SurfaceView, so installing it in fullscreen would contaminate the game render.
        if (!fullscreenRender) {
            backendLabel = new TextView(this);
            backendLabel.setTextColor(0xffffffff);
            backendLabel.setTextSize(15);
            backendLabel.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
            backendLabel.setPadding(dp(14), dp(7), dp(14), dp(7));
            backendLabel.setClickable(false);
            backendLabel.setFocusable(false);
            backendLabel.setElevation(dp(32));
            FrameLayout.LayoutParams backendParams = new FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP | Gravity.END);
            backendParams.rightMargin = dp(230);
            backendParams.topMargin = dp(16);
            getWindow().addContentView(backendLabel, backendParams);
            backendLabel.bringToFront();
            updateBackendLabel();
        }

        if (intentFlag(EXTRA_AUTOLOAD_EXISTING_RUNTIME) || isBareLaunch()
                || hasIntentAutomationExtras()) {
            root.postDelayed(() -> runIntentAutomation(root), AUTOMATION_START_DELAY_MS);
        }
        handleExternalControlIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        if (launcherMode) {
            setIntent(intent);
            if (isLauncherHomeIntent(intent) || !hasIntentAutomationExtras(intent)) {
                if (launcherUi != null) {
                    launcherUi.refresh();
                }
            } else {
                routeIntentToGameActivity(intent);
            }
            return;
        }
        Log.i(TAG, "Lifecycle onNewIntent action=" + intent.getAction());
        String incomingLaunchPath = intent.getStringExtra(EXTRA_LAUNCH_GAME_PATH);
        boolean incomingGameLaunch = incomingLaunchPath != null &&
                !incomingLaunchPath.trim().isEmpty();
        if (incomingGameLaunch && !activeGamePath.isEmpty()) {
            if (!activeGamePath.equals(incomingLaunchPath)) {
                Log.w(TAG, "Refusing second concurrent guest: active=" + activeGamePath +
                        " requested=" + incomingLaunchPath);
                Toast.makeText(this,
                        getString(R.string.another_game_active),
                        Toast.LENGTH_LONG).show();
            }
            // Bringing the existing singleTask GameActivity to front is sufficient for the same
            // title. Never start a second guest against process-global runtime state.
            return;
        }
        boolean wasFullscreen = fullscreenRender;
        setIntent(intent);
        boolean nextFullscreen = wantsFullscreenRender();
        if (wasFullscreen && !nextFullscreen) {
            nextFullscreen = true;
            append("Preserving fullscreen render mode across package-local resume intent.");
        }
        if (wasFullscreen != nextFullscreen) {
            // The Surface and diagnostic shell are different content trees. Window flags alone
            // cannot remove controls created by the old mode, so rebuild from the new Intent.
            fullscreenRender = nextFullscreen;
            recreate();
            return;
        }
        fullscreenRender = nextFullscreen;
        updateBackendLabel();
        if (fullscreenRender) {
            enableFullscreenRenderMode();
        }
        handleExternalControlIntent(intent);
        if (incomingGameLaunch && renderSurfaceView != null) {
            renderSurfaceView.postDelayed(() -> runIntentAutomation(renderSurfaceView), 100);
        }
    }

    private boolean isLauncherHomeIntent(Intent intent) {
        return intent != null
                && Intent.ACTION_MAIN.equals(intent.getAction())
                && intent.hasCategory(Intent.CATEGORY_LAUNCHER)
                && !hasIntentAutomationExtras();
    }

    private void routeIntentToGameActivity(Intent source) {
        Intent routed = source == null ? new Intent() : new Intent(source);
        routed.setClass(this, GameActivity.class);
        routed.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT |
                Intent.FLAG_ACTIVITY_SINGLE_TOP);
        // CATEGORY_LAUNCHER belongs to the portrait library, not the render host.
        routed.removeCategory(Intent.CATEGORY_LAUNCHER);
        startActivity(routed);
    }

    private void configurePortraitLauncherWindow() {
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN |
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);
        UiChrome.preparePortraitWindow(this);
    }

    private void updateBackendLabel() {
        if (backendLabel == null) {
            return;
        }
        boolean jitActive = false;
        String runtimeStatus = "";
        if (runtimeLoaded) {
            try {
                runtimeStatus = RuntimeBridge.jitStatus();
                jitActive = runtimeStatus != null &&
                        (runtimeStatus.contains("\"selectedBackend\":\"jit") ||
                                runtimeStatus.contains("\"name\":\"embedded-aarch64-jit-b\""));
            } catch (Throwable e) {
                Log.w(TAG, "Backend label runtime status unavailable", e);
            }
        }
        backendLabel.setBackgroundColor(0xee1565c0);
        backendLabel.setText(R.string.aarch64_jit);
        Log.i(TAG, "EXECUTOR_UI_JIT_LABEL active=" + jitActive +
                " shown=AArch64 JIT" +
                " runtimeStatus=" + runtimeStatus);
    }

    private void handleExternalControlIntent(Intent intent) {
        if (intent == null) {
            return;
        }
        if (smokeAction(ACTION_SMOKE_TEXT_SUFFIX).equals(intent.getAction())) {
            String text = intent.getStringExtra(EXTRA_SMOKE_TEXT_UTF8);
            if (text == null || text.isEmpty()) {
                append("Smoke text input ignored: missing text_utf8");
                return;
            }
            int result = queueSystemUiText(text, "smoke-intent");
            append(String.format(Locale.US,
                    "Smoke text input intent: {\"bytes\":%d,\"result\":%d}",
                    text.getBytes(StandardCharsets.UTF_8).length, result));
            return;
        }
        if (smokeAction(ACTION_SMOKE_PIXEL_COPY_SUFFIX).equals(intent.getAction())) {
            String reason = intent.getStringExtra("pixel_copy_reason");
            if (reason == null || reason.trim().isEmpty()) {
                reason = "external_live_probe";
            }
            requestSurfacePixelCopy(reason);
            append("Smoke PixelCopy intent accepted: {\"trigger\":\"" +
                    sanitizeJson(reason) + "\"}");
            return;
        }
        if (!smokeAction(ACTION_SMOKE_PAD_SUFFIX).equals(intent.getAction())) {
            return;
        }
        int mask = intent.getIntExtra(EXTRA_SMOKE_PAD_MASK, 0);
        if (mask == 0) {
            append("Smoke pad input ignored: missing mask");
            return;
        }
        int tapMs = intent.getIntExtra(EXTRA_SMOKE_PAD_TAP_MS, 0);
        if (tapMs > 0) {
            sendPadButton(mask, true);
            new Handler(Looper.getMainLooper()).postDelayed(
                    () -> sendPadButton(mask, false),
                    Math.max(32, Math.min(1200, tapMs)));
            append(String.format(Locale.US,
                    "Smoke pad tap intent: {\"mask\":\"0x%08x\",\"tapMs\":%d}", mask, tapMs));
            return;
        }
        boolean pressed = intent.getBooleanExtra(EXTRA_SMOKE_PAD_PRESSED, true);
        sendPadButton(mask, pressed);
        append(String.format(Locale.US,
                "Smoke pad input intent: {\"mask\":\"0x%08x\",\"pressed\":%s}",
                mask, pressed ? "true" : "false"));
    }

    private void runIntentAutomation(View root) {
        if (fullscreenRender && (currentSurface == null || !currentSurface.isValid())) {
            if (automationSurfaceWaitAttempts++ < 24) {
                append("Automation waiting for valid surface: attempt=" + automationSurfaceWaitAttempts);
                root.postDelayed(() -> runIntentAutomation(root), 250);
                return;
            }
            append("Automation continuing without valid surface after waits.");
        }
        automationSurfaceWaitAttempts = 0;

        Thread automationThread = new Thread(() -> runIntentAutomationSequence(root),
                "ps4run-intent-automation");
        automationThread.setDaemon(true);
        automationThread.start();
    }

    private void runIntentAutomationSequence(View root) {
        append("Automation extras: relocate=" + intentFlag(EXTRA_RELOCATE_IMPORTS_AFTER_LAUNCH) +
                " prepare=" + intentFlag(EXTRA_PREPARE_BOX64_ENTRY_AFTER_RELOCATE) +
                " validate=" + intentFlag(EXTRA_VALIDATE_EMBEDDED_BOX64_MAPPED_ENTRY) +
                " run=" + intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY) +
                " jit=" + intentFlag(EXTRA_EMBEDDED_AARCH64_JIT_BACKEND) +
                " fullscreen=" + intentFlag(EXTRA_FULLSCREEN_RENDER));

        boolean useTranslatorBridge = intentFlag(EXTRA_TRANSLATOR_BRIDGE_LAUNCH) ||
                intentFlag(EXTRA_JAVA_TRANSLATOR_SMOKE);
        boolean useLegacyBox64Diagnostics =
                intentFlag(EXTRA_VALIDATE_EMBEDDED_BOX64_MAPPED_ENTRY) ||
                intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY);
        loadRuntimeFromFile("Existing sandbox runtime");
        if (intentFlag(EXTRA_AUTOINIT_RUNTIME) || isBareLaunch() || hasIntentAutomationExtras()) {
            initializeRuntime();
        }
        String probeDlopenPath = getIntent().getStringExtra(EXTRA_PROBE_DLOPEN_PATH);
        if (probeDlopenPath != null && !probeDlopenPath.isEmpty()) {
            probeDlopenPath(probeDlopenPath);
        }
        String embeddedBox64Path = getIntent().getStringExtra(EXTRA_EMBEDDED_BOX64_PATH);
        if (useLegacyBox64Diagnostics &&
                (embeddedBox64Path == null || embeddedBox64Path.isEmpty())) {
            // Default-load the embedded box64 so the in-process embedded-ELF path (real guest
            // threading for SDL2/threading) is available without the smoke harness.
            java.io.File def = new java.io.File(getFilesDir(), "probes/libbox64_executor_embed.so");
            if (def.isFile()) embeddedBox64Path = def.getAbsolutePath();
        }
        if (useLegacyBox64Diagnostics &&
                embeddedBox64Path != null && !embeddedBox64Path.isEmpty()) {
            loadEmbeddedBox64Path(embeddedBox64Path);
        }
        String guestElfPath = getIntent().getStringExtra(EXTRA_TRANSLATOR_GUEST_ELF_PATH);
        if (guestElfPath != null && !guestElfPath.isEmpty()) {
            runTranslatorGuestElf(guestElfPath);
        }
        String scanPath = getIntent().getStringExtra(EXTRA_SCAN_GAME_PATH);
        if (scanPath != null && !scanPath.isEmpty()) {
            scanGamePath(scanPath);
        }
        String launchPath = getIntent().getStringExtra(EXTRA_LAUNCH_GAME_PATH);
        if (launchPath != null && !launchPath.isEmpty()) {
            activeGamePath = launchPath;
            if (useTranslatorBridge) {
                prepareTranslatorLaunchPath(launchPath);
            } else {
                launchGamePathJit(launchPath);
            }
            if (intentFlag(EXTRA_RELOCATE_IMPORTS_AFTER_LAUNCH) ||
                    intentFlag(EXTRA_PREPARE_BOX64_ENTRY_AFTER_RELOCATE) ||
                    intentFlag(EXTRA_VALIDATE_EMBEDDED_BOX64_MAPPED_ENTRY) ||
                    intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY)) {
                relocateImports();
                if ((intentFlag(EXTRA_PREPARE_BOX64_ENTRY_AFTER_RELOCATE) ||
                        intentFlag(EXTRA_VALIDATE_EMBEDDED_BOX64_MAPPED_ENTRY)) &&
                        !intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY)) {
                    prepareBox64EntryTrampoline();
                }
                if (intentFlag(EXTRA_VALIDATE_EMBEDDED_BOX64_MAPPED_ENTRY) &&
                        !intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY)) {
                    validateEmbeddedBox64MappedEntry();
                }
                if (intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY) || isLiveGameLaunch()) {
                    if (!intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY)) {
                        append("Mapped-entry run forced for live game launch.");
                    }
                    runEmbeddedBox64MappedEntry();
                }
                if (intentFlag(EXTRA_PRESENT_HOMEBREW_LOADER_FRAME)) {
                    root.postDelayed(this::presentHomebrewLoaderFrame, 250);
                }
            }
        }
        if (useTranslatorBridge) {
            runTranslatorBridgeHelper();
        }
        if (intentFlag(EXTRA_PRESENT_TEST_PATTERN) && !isLiveGameLaunch()) {
            root.postDelayed(this::presentTestPattern, 750);
        } else if (intentFlag(EXTRA_PRESENT_TEST_PATTERN)) {
            append("Present test pattern blocked for live game launch.");
        }
        if (intentFlag(EXTRA_PRESENT_FRAME_DUMP_ONLY)) {
            root.postDelayed(this::presentFrameDumpAfterGuest, 1000);
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (launcherMode) {
            return;
        }
        if (hasFocus && fullscreenRender) {
            enableFullscreenRenderMode();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (launcherMode) {
            configurePortraitLauncherWindow();
            if (launcherUi != null) {
                launcherUi.refresh();
            }
            return;
        }
        append("Lifecycle onResume fullscreen=" + fullscreenRender);
        configureLandscapeWindow();
        SharedPreferences appSettings = getSharedPreferences(SettingsActivity.PREFS, MODE_PRIVATE);
        applyLiveSetting(appSettings, null);
        registerInputDeviceListener();
        refreshExternalGamepadState();
        if (fullscreenRender) {
            enableFullscreenRenderMode();
            startSystemImeBridge();
        }
    }

    @Override
    protected void onPause() {
        if (launcherMode) {
            super.onPause();
            return;
        }
        append("Lifecycle onPause fullscreen=" + fullscreenRender);
        stopSystemImeBridge();
        resetExternalGamepadInputs();
        unregisterInputDeviceListener();
        super.onPause();
    }

    @Override
    protected void onStop() {
        if (launcherMode) {
            super.onStop();
            return;
        }
        append("Lifecycle onStop fullscreen=" + fullscreenRender);
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        if (launcherMode) {
            if (launcherUi != null) {
                launcherUi.destroy();
            }
            super.onDestroy();
            return;
        }
        append("Lifecycle onDestroy fullscreen=" + fullscreenRender);
        stopSystemImeBridge();
        systemImeHandler.removeCallbacks(systemImePoll);
        systemUiTextCapturedKeyCodes.clear();
        resetExternalGamepadInputs();
        unregisterInputDeviceListener();
        runtimeHudRunning = false;
        runtimeHudHandler.removeCallbacks(runtimeHudTick);
        if (liveSettings != null) {
            liveSettings.unregisterOnSharedPreferenceChangeListener(liveSettingsListener);
            liveSettings = null;
        }
        ToneGenerator tone = uiTone;
        uiTone = null;
        if (tone != null) {
            tone.release();
        }
        super.onDestroy();
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent event) {
        if (!launcherMode && event != null) {
            // Text gets first refusal only while a native modal owns it. This lets a hardware/adb
            // Space key type a space into sceImeDialog; without a modal it falls through to the
            // long-standing keyboard-as-Cross mapping below.
            if (handleSystemUiTextKey(event)) {
                return true;
            }
            if (event.getAction() != KeyEvent.ACTION_DOWN
                    && event.getAction() != KeyEvent.ACTION_UP) {
                return super.dispatchKeyEvent(event);
            }
            // Intercept before Window/View focus navigation. DPAD_LEFT/RIGHT otherwise may be
            // consumed by a focusable Surface/View and never reach Activity.onKeyDown(). This is
            // the same early-dispatch shape used by mature SDL-based Android emulator frontends.
            boolean pressed = event.getAction() == KeyEvent.ACTION_DOWN;
            if (handlePadKey(event.getKeyCode(), pressed, event)) {
                return true;
            }
        }
        return super.dispatchKeyEvent(event);
    }

    private boolean handleSystemUiTextKey(KeyEvent event) {
        int action = event.getAction();
        int keyCode = event.getKeyCode();

        // A printable key-down consumed as text must also own its matching key-up. Otherwise Space
        // would leak a synthetic PAD_CROSS release to the guest after being inserted into the IME.
        if (action == KeyEvent.ACTION_UP) {
            return systemUiTextCapturedKeyCodes.remove(keyCode);
        }
        if (action != KeyEvent.ACTION_DOWN && action != KeyEvent.ACTION_MULTIPLE) {
            return false;
        }

        InputDevice device = event.getDevice();
        boolean controllerEvent = isControllerSource(event.getSource())
                || (device != null && isControllerSource(device.getSources()));
        if (controllerEvent) {
            return false;
        }

        String text;
        if (action == KeyEvent.ACTION_MULTIPLE) {
            text = event.getCharacters();
        } else {
            int codePoint = event.getUnicodeChar(event.getMetaState());
            if (!Character.isValidCodePoint(codePoint) || Character.isISOControl(codePoint)) {
                return false;
            }
            text = new String(Character.toChars(codePoint));
        }
        if (!isPrintableSystemUiText(text)) {
            return false;
        }

        int result = queueSystemUiText(text, "hardware-key");
        if (result <= 0) {
            return false;
        }
        if (action == KeyEvent.ACTION_DOWN && keyCode != KeyEvent.KEYCODE_UNKNOWN) {
            systemUiTextCapturedKeyCodes.add(keyCode);
        }
        return true;
    }

    private boolean isPrintableSystemUiText(String text) {
        if (text == null || text.isEmpty()) {
            return false;
        }
        for (int offset = 0; offset < text.length();) {
            int codePoint = text.codePointAt(offset);
            if (!Character.isValidCodePoint(codePoint) || Character.isISOControl(codePoint)) {
                return false;
            }
            offset += Character.charCount(codePoint);
        }
        return true;
    }

    private int queueSystemUiText(String text, String source) {
        int byteCount = text == null ? 0 : text.getBytes(StandardCharsets.UTF_8).length;
        try {
            int result = RuntimeBridge.sendSystemUiText(text);
            String line = String.format(Locale.US,
                    "EXECUTOR_UI_SYSTEM_TEXT source=%s bytes=%d result=%d",
                    source, byteCount, result);
            Log.i(TAG, line);
            appendSmokeLog(line);
            runtimeHudEvent = String.format(Locale.US,
                    "IME text %s: %d bytes result=%d", source, byteCount, result);
            return result;
        } catch (Throwable t) {
            Log.w(TAG, "EXECUTOR_UI_SYSTEM_TEXT failed source=" + source +
                    " bytes=" + byteCount, t);
            runtimeHudEvent = "IME text bridge failed";
            return -4;
        }
    }

    private void installSystemImeBridge(FrameLayout host) {
        if (systemImeEditText != null || host == null) {
            return;
        }

        EditText editText = new EditText(this);
        systemImeEditText = editText;
        editText.setSingleLine(true);
        editText.setMaxLines(1);
        editText.setInputType(InputType.TYPE_CLASS_TEXT |
                InputType.TYPE_TEXT_FLAG_CAP_SENTENCES |
                InputType.TYPE_TEXT_FLAG_AUTO_CORRECT);
        editText.setImeOptions(EditorInfo.IME_ACTION_DONE |
                EditorInfo.IME_FLAG_NO_EXTRACT_UI |
                EditorInfo.IME_FLAG_NO_FULLSCREEN);
        editText.setFilters(new InputFilter[]{new InputFilter.LengthFilter(4096)});
        editText.setBackgroundColor(0x00000000);
        editText.setTextColor(0x00000000);
        editText.setHintTextColor(0x00000000);
        editText.setCursorVisible(false);
        editText.setAlpha(0.01f);
        editText.setPadding(0, 0, 0, 0);
        editText.setSaveEnabled(false);
        editText.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO);

        editText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence text, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence text, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable editable) {
                if (!systemImeInternalEdit && systemImeVisible) {
                    forwardCommittedSystemImeText(editable);
                }
            }
        });
        editText.setOnEditorActionListener((view, actionId, event) -> {
            boolean done = actionId == EditorInfo.IME_ACTION_DONE ||
                    (event != null && event.getKeyCode() == KeyEvent.KEYCODE_ENTER &&
                            event.getAction() == KeyEvent.ACTION_DOWN);
            if (!done) {
                return false;
            }
            int result;
            try {
                result = RuntimeBridge.submitSystemUiText();
            } catch (Throwable t) {
                Log.w(TAG, "EXECUTOR_UI_SYSTEM_TEXT submit failed", t);
                result = -4;
            }
            Log.i(TAG, "EXECUTOR_UI_SYSTEM_TEXT source=soft-ime-done result=" + result);
            runtimeHudEvent = "IME Done result=" + result;
            if (result > 0) {
                // Do not reopen during the renderer frame between queuing Enter and closing the
                // modal. If the native queue rejected Enter, keep the keyboard open so a transient
                // active-state race cannot strand the still-open sceImeDialog without input.
                systemImeSuppressUntilInactive = true;
                hideSystemIme();
                return true;
            }
            return false;
        });

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(1, 1,
                Gravity.BOTTOM | Gravity.START);
        host.addView(editText, params);
    }

    private void startSystemImeBridge() {
        if (systemImeEditText == null || systemImeBridgeRunning) {
            return;
        }
        systemImeBridgeRunning = true;
        systemImeHandler.removeCallbacks(systemImePoll);
        systemImeHandler.post(systemImePoll);
    }

    private void stopSystemImeBridge() {
        systemImeBridgeRunning = false;
        systemImeHandler.removeCallbacks(systemImePoll);
        systemImeSuppressUntilInactive = false;
        hideSystemIme();
    }

    private void updateSystemImeVisibility() {
        boolean active = false;
        if (runtimeLoaded && fullscreenRender && systemImeEditText != null) {
            try {
                active = RuntimeBridge.isSystemUiTextActive() > 0;
            } catch (Throwable t) {
                Log.w(TAG, "EXECUTOR_UI_SYSTEM_TEXT active query failed", t);
            }
        }

        if (!active) {
            systemImeSuppressUntilInactive = false;
            if (systemImeVisible) {
                hideSystemIme();
            }
            return;
        }
        if (!systemImeVisible && !systemImeSuppressUntilInactive) {
            showSystemIme();
        }
    }

    private void showSystemIme() {
        EditText editText = systemImeEditText;
        if (editText == null || systemImeVisible || !hasWindowFocus()) {
            return;
        }
        resetSystemImeBuffer();
        systemImeVisible = true;
        editText.setVisibility(View.VISIBLE);
        editText.requestFocus();
        editText.post(() -> {
            if (!systemImeVisible || systemImeEditText != editText) {
                return;
            }
            InputMethodManager inputMethodManager =
                    (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
            if (inputMethodManager != null) {
                inputMethodManager.restartInput(editText);
                boolean shown = inputMethodManager.showSoftInput(
                        editText, InputMethodManager.SHOW_IMPLICIT);
                Log.i(TAG, "EXECUTOR_UI_SYSTEM_TEXT soft-ime-show=" + shown);
                if (!shown) {
                    // A request issued on the first focus/layout turn is allowed to be rejected by
                    // InputMethodManager. Keep the logical IME session alive and retry after the
                    // editor has an attached, focused window; otherwise one transient rejection
                    // strands a still-active sceImeDialog behind an invisible 1x1 editor.
                    editText.postDelayed(() -> {
                        if (systemImeVisible && systemImeEditText == editText &&
                                editText.hasWindowFocus() && editText.hasFocus()) {
                            boolean retried = inputMethodManager.showSoftInput(
                                    editText, InputMethodManager.SHOW_IMPLICIT);
                            Log.i(TAG, "EXECUTOR_UI_SYSTEM_TEXT soft-ime-retry=" + retried);
                        }
                    }, 160);
                }
            }
        });
    }

    private void hideSystemIme() {
        EditText editText = systemImeEditText;
        if (editText == null) {
            systemImeVisible = false;
            return;
        }
        InputMethodManager inputMethodManager =
                (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (inputMethodManager != null) {
            inputMethodManager.hideSoftInputFromWindow(editText.getWindowToken(), 0);
        }
        systemImeVisible = false;
        editText.clearFocus();
        resetSystemImeBuffer();
        View decor = getWindow().getDecorView();
        if (decor != null) {
            decor.requestFocus();
        }
    }

    private void resetSystemImeBuffer() {
        EditText editText = systemImeEditText;
        if (editText == null) {
            systemImeCommittedText = "";
            return;
        }
        systemImeInternalEdit = true;
        editText.setText("");
        editText.setSelection(0);
        systemImeCommittedText = "";
        systemImeInternalEdit = false;
    }

    private void forwardCommittedSystemImeText(Editable editable) {
        if (editable == null) {
            return;
        }
        int composingStart = BaseInputConnection.getComposingSpanStart(editable);
        if (composingStart >= 0) {
            // Composition is provisional. Only a newly committed prefix before it is safe to send;
            // the composing span itself is delivered once the IME commits/removes that span.
            String committedPrefix = editable.subSequence(0, composingStart).toString();
            if (committedPrefix.startsWith(systemImeCommittedText) &&
                    committedPrefix.length() > systemImeCommittedText.length()) {
                String appended = committedPrefix.substring(systemImeCommittedText.length());
                if (queueSystemUiText(appended, "soft-ime-commit") > 0) {
                    systemImeCommittedText = committedPrefix;
                }
            }
            return;
        }

        String current = editable.toString();
        String previous = systemImeCommittedText;
        int commonPrefix = commonUtf16Prefix(previous, current);
        if (commonPrefix < previous.length()) {
            int removedCodePoints = previous.codePointCount(commonPrefix, previous.length());
            int result;
            try {
                result = RuntimeBridge.backspaceSystemUiText(removedCodePoints);
            } catch (Throwable t) {
                Log.w(TAG, "EXECUTOR_UI_SYSTEM_TEXT backspace failed count=" +
                        removedCodePoints, t);
                return;
            }
            if (result <= 0) {
                return;
            }
            systemImeCommittedText = previous.substring(0, commonPrefix);
            previous = systemImeCommittedText;
        }
        if (commonPrefix < current.length()) {
            String appended = current.substring(commonPrefix);
            if (queueSystemUiText(appended, "soft-ime-commit") <= 0) {
                return;
            }
        }
        systemImeCommittedText = current;
    }

    private int commonUtf16Prefix(String left, String right) {
        int limit = Math.min(left.length(), right.length());
        int index = 0;
        while (index < limit && left.charAt(index) == right.charAt(index)) {
            index++;
        }
        // Never split a supplementary Unicode code point when translating a replacement into
        // ImGui backspaces plus newly committed UTF-8.
        if (index > 0 && index < left.length() && index < right.length() &&
                Character.isHighSurrogate(left.charAt(index - 1))) {
            index--;
        }
        return index;
    }

    private boolean wantsFullscreenRender() {
        // GameActivity has one UI contract: fullscreen PS4 output plus the optional controller
        // overlay.  The obsolete grey diagnostic button shell no longer has an Intent route.
        return true;
    }

    // A "bare" launch = tapping the app icon with no automation extras. In that
    // case bring the runtime up automatically so the user lands directly on a
    // live fullscreen render instead of a black surface.
    private boolean isBareLaunch() {
        Intent intent = getIntent();
        if (intent != null && (smokeAction(ACTION_SMOKE_PAD_SUFFIX).equals(intent.getAction()) ||
                smokeAction(ACTION_SMOKE_TEXT_SUFFIX).equals(intent.getAction()) ||
                smokeAction(ACTION_SMOKE_PIXEL_COPY_SUFFIX).equals(intent.getAction()))) {
            return false;
        }
        return !hasIntentAutomationExtras();
    }

    private boolean hasIntentAutomationExtras() {
        return hasIntentAutomationExtras(getIntent());
    }

    private String smokeAction(String suffix) {
        return getPackageName() + suffix;
    }

    private boolean hasIntentAutomationExtras(Intent intent) {
        String action = intent == null ? null : intent.getAction();
        return smokeAction(ACTION_SMOKE_PAD_SUFFIX).equals(action)
                || smokeAction(ACTION_SMOKE_TEXT_SUFFIX).equals(action)
                || smokeAction(ACTION_SMOKE_PIXEL_COPY_SUFFIX).equals(action)
                || intentFlag(intent, EXTRA_AUTOLOAD_EXISTING_RUNTIME)
                || intentFlag(intent, EXTRA_AUTOINIT_RUNTIME)
                || intentFlag(intent, EXTRA_FULLSCREEN_RENDER)
                || intentFlag(intent, EXTRA_EMBEDDED_AARCH64_JIT_BACKEND)
                || intentFlag(intent, EXTRA_PRESENT_TEST_PATTERN)
                || intentFlag(intent, EXTRA_PRESENT_HOMEBREW_LOADER_FRAME)
                || intentFlag(intent, EXTRA_PRESENT_FRAME_DUMP_ONLY)
                || intentFlag(intent, EXTRA_PRESENT_FRAME_DUMP_AFTER_GUEST)
                || intentFlag(intent, EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY)
                || intentFlag(intent, EXTRA_TRANSLATOR_BRIDGE_LAUNCH)
                || intentFlag(intent, EXTRA_JAVA_TRANSLATOR_SMOKE)
                || (intent != null && intent.getStringExtra(EXTRA_PROBE_DLOPEN_PATH) != null)
                || (intent != null && intent.getStringExtra(EXTRA_EMBEDDED_BOX64_PATH) != null)
                || (intent != null && intent.getStringExtra(EXTRA_LAUNCH_GAME_PATH) != null)
                || (intent != null && intent.getStringExtra(EXTRA_SCAN_GAME_PATH) != null)
                || (intent != null && intent.getStringExtra(EXTRA_TRANSLATOR_GUEST_ELF_PATH) != null);
    }

    private boolean isLiveGameLaunch() {
        Intent intent = getIntent();
        if (intent == null) {
            return false;
        }
        String launchPath = intent.getStringExtra(EXTRA_LAUNCH_GAME_PATH);
        if (launchPath == null || launchPath.isEmpty()) {
            return false;
        }
        return intent.getBooleanExtra(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY, false)
                || launchPath.contains("/CUSA")
                || launchPath.endsWith("/eboot.bin");
    }

    private boolean intentFlag(String name) {
        return intentFlag(getIntent(), name);
    }

    private boolean intentFlag(Intent intent, String name) {
        if (intent == null || name == null) {
            return false;
        }
        if (intent.getBooleanExtra(name, false)) {
            return true;
        }
        Bundle extras = intent.getExtras();
        if (extras == null || !extras.containsKey(name)) {
            return false;
        }
        Object value = extras.get(name);
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof String) {
            String text = ((String) value).trim();
            return "true".equalsIgnoreCase(text) || "1".equals(text) || "yes".equalsIgnoreCase(text);
        }
        if (value instanceof Number) {
            return ((Number) value).intValue() != 0;
        }
        return false;
    }

    private boolean shouldShowStoreCatalogPreview() {
        // HARD-DISABLED (authentic-only, reaffirmed 2026-05-31): absolutely no hardcode / no
        // Android-layout facsimile. The PS4 Store must render fully and authentically as the guest
        // ELF via Piglet/EGL — never a host-drawn StoreCatalogView. The catalog facsimile (tabs /
        // detail / filtering wiring) is kept dormant in this file for reference only and must never
        // draw. The real work is fixing the guest ELF font/text render (silent SIGKILL at the
        // freetype glyph-rasterization stage in on_GLES2_Init_icons), not faking it on Android.
        return false;
    }

    private void configureLandscapeWindow() {
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().setFormat(PixelFormat.RGBA_8888);
        getWindow().setLayout(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT);
        // Keep the 16:9 SurfaceView attached and sized while the landscape soft keyboard overlays
        // it; resizing/recreating the render host here can abandon in-flight Vulkan presents.
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_NOTHING |
                WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN);
        getWindow().addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON |
                        WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON |
                        WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED |
                        WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setTurnScreenOn(true);
            setShowWhenLocked(true);
            KeyguardManager keyguardManager = (KeyguardManager) getSystemService(Context.KEYGUARD_SERVICE);
            if (keyguardManager != null) {
                keyguardManager.requestDismissKeyguard(this, null);
            }
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            WindowManager.LayoutParams params = getWindow().getAttributes();
            params.layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            getWindow().setAttributes(params);
        }
    }

    private void enableFullscreenRenderMode() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            getWindow().setDecorFitsSystemWindows(false);
            View decor = getWindow().getDecorView();
            WindowInsetsController controller = decor != null ? decor.getWindowInsetsController() : null;
            if (controller != null) {
                controller.hide(WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
                controller.setSystemBarsBehavior(
                        WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
        }
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                        View.SYSTEM_UI_FLAG_FULLSCREEN |
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                        View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
                        View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |
                        View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    private void registerInputDeviceListener() {
        if (inputManager == null) {
            inputManager = (InputManager) getSystemService(Context.INPUT_SERVICE);
        }
        if (inputManager != null && !inputDeviceListenerRegistered) {
            inputManager.registerInputDeviceListener(inputDeviceListener, null);
            inputDeviceListenerRegistered = true;
        }
    }

    private void unregisterInputDeviceListener() {
        if (inputManager != null && inputDeviceListenerRegistered) {
            inputManager.unregisterInputDeviceListener(inputDeviceListener);
            inputDeviceListenerRegistered = false;
        }
    }

    private boolean isControllerSource(int sources) {
        return (sources & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD
                || (sources & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK
                || (sources & InputDevice.SOURCE_DPAD) == InputDevice.SOURCE_DPAD;
    }

    private boolean isExternalGamepad(InputDevice device) {
        return "eligible".equals(gamepadEligibility(device));
    }

    private String gamepadEligibility(InputDevice device) {
        if (device == null) {
            return "null-device";
        }
        if (!isControllerSource(device.getSources())) {
            return "not-controller-source";
        }
        boolean gamepadOrJoystick = (device.getSources() & InputDevice.SOURCE_GAMEPAD)
                == InputDevice.SOURCE_GAMEPAD
                || (device.getSources() & InputDevice.SOURCE_JOYSTICK)
                == InputDevice.SOURCE_JOYSTICK;
        if (!gamepadOrJoystick
                && device.getKeyboardType() == InputDevice.KEYBOARD_TYPE_ALPHABETIC) {
            return "alphabetic-dpad-keyboard";
        }
        if (device.isVirtual()) {
            return "android-virtual";
        }
        String name = device.getName() == null ? "" : device.getName().toLowerCase(Locale.ROOT);
        // Xiaomi exposes an OEM uinput bridge as GAMEPAD (0666:0888) while no physical pad is
        // attached. Android's InputDevice.isVirtual() is false for it, so the old predicate hid
        // the touch controls permanently. Reject software/uinput controller bridges explicitly.
        if (name.startsWith("uinput-") || name.contains("virtual")
                || (device.getVendorId() == 0x0666 && device.getProductId() == 0x0888)) {
            return "software-uinput";
        }
        return "eligible";
    }

    private void logGamepadDevice(InputDevice device, String eligibility) {
        if (device == null || !isControllerSource(device.getSources())) {
            return;
        }
        StringBuilder ranges = new StringBuilder();
        for (InputDevice.MotionRange range : device.getMotionRanges()) {
            if (!isControllerSource(range.getSource())) {
                continue;
            }
            if (ranges.length() > 0) ranges.append(';');
            ranges.append(MotionEvent.axisToString(range.getAxis()))
                    .append('[').append(range.getMin()).append(',').append(range.getMax())
                    .append(" flat=").append(range.getFlat()).append(']');
        }
        Log.i(TAG, String.format(Locale.US,
                "EXECUTOR_GAMEPAD_DEVICE id=%d name=%s descriptor=%s vid=%04x pid=%04x " +
                        "sources=0x%08x virtual=%s eligibility=%s ranges=%s",
                device.getId(), device.getName(), device.getDescriptor(), device.getVendorId(),
                device.getProductId(), device.getSources(), device.isVirtual(), eligibility,
                ranges));
    }

    private void refreshExternalGamepadState() {
        boolean connected = false;
        String name = "";
        int deviceId = -1;
        String descriptor = "";
        for (int id : InputDevice.getDeviceIds()) {
            InputDevice device = InputDevice.getDevice(id);
            String eligibility = gamepadEligibility(device);
            logGamepadDevice(device, eligibility);
            if (!connected && "eligible".equals(eligibility)) {
                connected = true;
                name = device.getName();
                deviceId = device.getId();
                descriptor = device.getDescriptor();
            }
        }
        boolean disconnected = externalGamepadConnected && !connected;
        boolean changed = connected != externalGamepadConnected
                || deviceId != externalGamepadDeviceId
                || !String.valueOf(descriptor).equals(externalGamepadDescriptor);
        boolean switchedDevice = externalGamepadConnected && connected
                && (deviceId != externalGamepadDeviceId
                || !String.valueOf(descriptor).equals(externalGamepadDescriptor));
        if (disconnected || switchedDevice) {
            resetExternalGamepadInputs();
        }
        externalGamepadConnected = connected;
        externalGamepadName = name == null ? "" : name;
        externalGamepadDeviceId = deviceId;
        externalGamepadDescriptor = descriptor == null ? "" : descriptor;
        updateTouchOverlayVisibility();
        if (connected && changed) {
            runtimeHudEvent = getString(R.string.hud_external_controller,
                    externalGamepadName);
        } else if (disconnected) {
            runtimeHudEvent = getString(R.string.hud_controller_disconnected);
        }
    }

    private void noteExternalGamepadEvent(InputDevice device) {
        if (!isExternalGamepad(device)) {
            return;
        }
        if (!externalGamepadConnected) {
            externalGamepadConnected = true;
            externalGamepadName = device == null ? "Controller" : device.getName();
            externalGamepadDeviceId = device == null ? -1 : device.getId();
            externalGamepadDescriptor = device == null ? "" : device.getDescriptor();
            updateTouchOverlayVisibility();
        }
    }

    private void updateTouchOverlayVisibility() {
        boolean show = overlayEnabledByPreference && !externalGamepadConnected;
        if (lastTouchOverlayVisible == null || lastTouchOverlayVisible != show) {
            String reason = !overlayEnabledByPreference ? "setting-disabled"
                    : externalGamepadConnected ? "external-gamepad" : "enabled";
            Log.i(TAG, "EXECUTOR_TOUCH_OVERLAY visible=" + show
                    + " preference=" + overlayEnabledByPreference
                    + " externalGamepad=" + externalGamepadConnected
                    + " deviceId=" + externalGamepadDeviceId
                    + " controller=" + externalGamepadName
                    + " reason=" + reason);
            lastTouchOverlayVisible = show;
        }
        if (inputOverlay == null) {
            return;
        }
        if (!show && inputOverlay.getVisibility() != View.GONE && dualShockOverlay != null) {
            dualShockOverlay.resetAllInputs();
        }
        inputOverlay.setVisibility(show ? View.VISIBLE : View.GONE);
    }

    private void resetExternalGamepadInputs() {
        externalButtonMask = 0;
        externalHatMask = 0;
        applyExternalDigitalMask();
        int[] neutral = {128, 128, 128, 128, 0, 0};
        for (int axis = 0; axis < neutral.length; axis++) {
            externalAxisZones[axis] = 0;
            if (externalAxisValues[axis] != neutral[axis]) {
                externalAxisValues[axis] = neutral[axis];
                if (runtimeLoaded) {
                    sendPadAxis(axis, neutral[axis], false);
                }
            }
        }
    }

    private void applyExternalDigitalMask() {
        int next = externalButtonMask | externalHatMask;
        int changed = externalSentMask ^ next;
        if (changed == 0) {
            return;
        }
        if (runtimeLoaded) {
            // Release first, press second. On a direct LEFT->RIGHT hat transition both bits change
            // in one MotionEvent; the old single pass briefly pressed RIGHT while LEFT was still
            // down. A guest polling in that window saw an impossible two-direction state.
            for (int bit : PAD_DIGITAL_BITS) {
                if ((changed & bit) != 0 && (next & bit) == 0) {
                    sendPadButton(bit, false, false);
                }
            }
            for (int bit : PAD_DIGITAL_BITS) {
                if ((changed & bit) != 0 && (next & bit) != 0) {
                    sendPadButton(bit, true, false);
                }
            }
        }
        externalSentMask = next;
    }

    private void replayExternalGamepadState() {
        if (!runtimeLoaded || !externalGamepadConnected) {
            return;
        }
        externalSentMask = 0;
        applyExternalDigitalMask();
        for (int axis = 0; axis < externalAxisValues.length; axis++) {
            sendPadAxis(axis, externalAxisValues[axis], false);
        }
    }

    @Override
    public boolean dispatchGenericMotionEvent(MotionEvent event) {
        if (!launcherMode && handleExternalGamepadMotion(event)) {
            return true;
        }
        return super.dispatchGenericMotionEvent(event);
    }

    private boolean handleExternalGamepadMotion(MotionEvent event) {
        if (event == null || event.getActionMasked() != MotionEvent.ACTION_MOVE
                || !isExternalGamepad(event.getDevice())) {
            return false;
        }
        boolean joystickSource = (event.getSource() & InputDevice.SOURCE_JOYSTICK)
                == InputDevice.SOURCE_JOYSTICK;
        boolean dpadSource = (event.getSource() & InputDevice.SOURCE_DPAD)
                == InputDevice.SOURCE_DPAD;
        if (!joystickSource && !dpadSource) {
            // Composite DualSense-compatible devices also emit touchpad/mouse/sensor MotionEvents.
            // Never interpret those AXIS_X/Y coordinates as centered joystick samples.
            return false;
        }
        InputDevice device = event.getDevice();
        noteExternalGamepadEvent(device);
        if (externalGamepadDeviceId >= 0 && device.getId() != externalGamepadDeviceId) {
            // One guest pad is exposed. Keep another connected controller from releasing or
            // replacing the selected controller's state.
            return true;
        }

        // Android batches joystick movements. Process oldest -> newest so quick left/right taps
        // are not lost between frames, then process the current sample at historyPos=-1.
        for (int historyPos = 0; historyPos < event.getHistorySize(); historyPos++) {
            processExternalGamepadMotionSample(event, historyPos, joystickSource);
        }
        processExternalGamepadMotionSample(event, -1, joystickSource);
        return true;
    }

    private void processExternalGamepadMotionSample(MotionEvent event, int historyPos,
                                                    boolean joystickSource) {
        InputDevice device = event.getDevice();
        if (joystickSource) {
            if (hasAxisPair(event, MotionEvent.AXIS_X, MotionEvent.AXIS_Y)) {
                sendExternalAxisIfChanged(0,
                        readStickAxis(event, MotionEvent.AXIS_X, historyPos),
                        MotionEvent.AXIS_X);
                sendExternalAxisIfChanged(1,
                        readStickAxis(event, MotionEvent.AXIS_Y, historyPos),
                        MotionEvent.AXIS_Y);
            }

            int rightX = -1;
            int rightY = -1;
            if (hasBipolarAxisPair(event, MotionEvent.AXIS_Z, MotionEvent.AXIS_RZ)) {
                rightX = MotionEvent.AXIS_Z;
                rightY = MotionEvent.AXIS_RZ;
            } else if (hasBipolarAxisPair(event, MotionEvent.AXIS_RX, MotionEvent.AXIS_RY)) {
                rightX = MotionEvent.AXIS_RX;
                rightY = MotionEvent.AXIS_RY;
            }
            if (rightX >= 0) {
                sendExternalAxisIfChanged(2, readStickAxis(event, rightX, historyPos), rightX);
                sendExternalAxisIfChanged(3, readStickAxis(event, rightY, historyPos), rightY);
            }

            int leftTrigger = hasAxis(event, MotionEvent.AXIS_LTRIGGER)
                    ? MotionEvent.AXIS_LTRIGGER : MotionEvent.AXIS_BRAKE;
            int rightTrigger = hasAxis(event, MotionEvent.AXIS_RTRIGGER)
                    ? MotionEvent.AXIS_RTRIGGER : MotionEvent.AXIS_GAS;
            if (hasAxis(event, leftTrigger)) {
                sendExternalAxisIfChanged(4,
                        readTriggerAxis(event, leftTrigger, historyPos), leftTrigger);
            }
            if (hasAxis(event, rightTrigger)) {
                sendExternalAxisIfChanged(5,
                        readTriggerAxis(event, rightTrigger, historyPos), rightTrigger);
            }
        }

        float hatX = readRawAxis(event, MotionEvent.AXIS_HAT_X, historyPos);
        float hatY = readRawAxis(event, MotionEvent.AXIS_HAT_Y, historyPos);
        int nextHat = 0;
        if (hatX <= -0.5f) nextHat |= PAD_LEFT;
        if (hatX >= 0.5f) nextHat |= PAD_RIGHT;
        if (hatY <= -0.5f) nextHat |= PAD_UP;
        if (hatY >= 0.5f) nextHat |= PAD_DOWN;
        if (nextHat != externalHatMask) {
            externalHatMask = nextHat;
            applyExternalDigitalMask();
        }
    }

    private InputDevice.MotionRange motionRange(MotionEvent event, int axis) {
        InputDevice device = event == null ? null : event.getDevice();
        if (device == null) {
            return null;
        }
        InputDevice.MotionRange range = device.getMotionRange(axis, event.getSource());
        if (range == null) {
            range = device.getMotionRange(axis);
        }
        return range;
    }

    private boolean hasAxis(MotionEvent event, int axis) {
        return motionRange(event, axis) != null;
    }

    private boolean hasAxisPair(MotionEvent event, int first, int second) {
        return hasAxis(event, first) && hasAxis(event, second);
    }

    private boolean hasBipolarAxisPair(MotionEvent event, int first, int second) {
        InputDevice.MotionRange firstRange = motionRange(event, first);
        InputDevice.MotionRange secondRange = motionRange(event, second);
        return firstRange != null && secondRange != null
                && firstRange.getMin() < 0f && firstRange.getMax() > 0f
                && secondRange.getMin() < 0f && secondRange.getMax() > 0f;
    }

    private float readRawAxis(MotionEvent event, int axis, int historyPos) {
        return historyPos < 0 ? event.getAxisValue(axis)
                : event.getHistoricalAxisValue(axis, historyPos);
    }

    private int readStickAxis(MotionEvent event, int axis, int historyPos) {
        InputDevice.MotionRange range = motionRange(event, axis);
        if (range == null) {
            return 128;
        }
        float value = readRawAxis(event, axis, historyPos);
        float center = (range.getMin() + range.getMax()) * 0.5f;
        float halfRange = Math.max(0.0001f, (range.getMax() - range.getMin()) * 0.5f);
        float normalized = (value - center) / halfRange;
        if (Math.abs(value - center) <= Math.max(range.getFlat(), halfRange * 0.04f)) {
            normalized = 0f;
        }
        normalized = Math.max(-1f, Math.min(1f, normalized));
        return Math.max(0, Math.min(255, Math.round(128f + normalized * 127f)));
    }

    private int readTriggerAxis(MotionEvent event, int axis, int historyPos) {
        InputDevice.MotionRange range = motionRange(event, axis);
        if (range == null || range.getMax() <= range.getMin()) {
            return 0;
        }
        float normalized = (readRawAxis(event, axis, historyPos) - range.getMin()) /
                (range.getMax() - range.getMin());
        if (normalized <= 0.04f) {
            normalized = 0f;
        }
        return Math.max(0, Math.min(255, Math.round(normalized * 255f)));
    }

    private void sendExternalAxisIfChanged(int axis, int value, int sourceAxis) {
        if (axis < 0 || axis >= externalAxisValues.length ||
                Math.abs(externalAxisValues[axis] - value) <= 2) {
            return;
        }
        externalAxisValues[axis] = value;
        int zone = axis < 4 ? (value < 64 ? -1 : value > 192 ? 1 : 0)
                : (value <= 16 ? 0 : value >= 192 ? 1 : 2);
        if (externalAxisZones[axis] != zone) {
            externalAxisZones[axis] = zone;
            Log.i(TAG, "EXECUTOR_GAMEPAD_AXIS deviceId=" + externalGamepadDeviceId
                    + " slot=" + axis + " source=" + MotionEvent.axisToString(sourceAxis)
                    + " value=" + value + " zone=" + zone);
        }
        if (runtimeLoaded) {
            sendPadAxis(axis, value, false);
        }
    }

    private void installGamepadOverlay(FrameLayout renderFrame) {
        inputOverlay = new FrameLayout(this);
        inputOverlay.setFocusable(false);
        inputOverlay.setClickable(false);
        renderFrame.addView(inputOverlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
                Gravity.CENTER));

        dualShockOverlay = new DualShockOverlayView(this);
        inputOverlay.addView(dualShockOverlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
                Gravity.CENTER));

        inputOverlay.bringToFront();
        updateTouchOverlayVisibility();
        append("Input overlay installed: DualShock 4 visual pad, multitouch sticks, touchpad.");
    }

    private void installLoadingProgress(FrameLayout renderFrame) {
        if (loadingProgressPanel != null || renderFrame == null) {
            return;
        }
        loadingProgressPanel = new LinearLayout(this);
        loadingProgressPanel.setOrientation(LinearLayout.VERTICAL);
        loadingProgressPanel.setGravity(Gravity.CENTER_HORIZONTAL);
        loadingProgressPanel.setClickable(false);
        loadingProgressPanel.setFocusable(false);
        loadingProgressPanel.setElevation(dp(44));

        loadingBlocksText = new TextView(this);
        loadingBlocksText.setText("blocks 0");
        loadingBlocksText.setTextColor(0xff55e37a);
        loadingBlocksText.setTextSize(9);
        loadingBlocksText.setTypeface(Typeface.create("sans-serif", Typeface.NORMAL));
        loadingBlocksText.setGravity(Gravity.CENTER);
        loadingBlocksText.setIncludeFontPadding(false);
        loadingProgressPanel.addView(loadingBlocksText, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        ThinSpinnerView progress = new ThinSpinnerView(this);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                dp(24), dp(24));
        progressParams.gravity = Gravity.CENTER_HORIZONTAL;
        progressParams.topMargin = dp(4);
        loadingProgressPanel.addView(progress, progressParams);

        FrameLayout.LayoutParams panelParams = new FrameLayout.LayoutParams(
                dp(150), ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.END | Gravity.CENTER_VERTICAL);
        panelParams.rightMargin = dp(20);
        renderFrame.addView(loadingProgressPanel, panelParams);
        loadingProgressPanel.bringToFront();
        resetRuntimeDebugHudForGuestLaunch();
        startRuntimeDebugHud();
    }

    private static final class ThinSpinnerView extends View {
        private static final long REVOLUTION_MS = 900;
        private final Paint arcPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final RectF arcBounds = new RectF();

        ThinSpinnerView(Context context) {
            super(context);
            arcPaint.setColor(0xffffffff);
            arcPaint.setStyle(Paint.Style.STROKE);
            arcPaint.setStrokeCap(Paint.Cap.ROUND);
            arcPaint.setStrokeWidth(context.getResources().getDisplayMetrics().density);
            setClickable(false);
            setFocusable(false);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            final float inset = arcPaint.getStrokeWidth();
            arcBounds.set(inset, inset, getWidth() - inset, getHeight() - inset);
            final float start = (SystemClock.uptimeMillis() % REVOLUTION_MS) *
                    (360f / REVOLUTION_MS);
            canvas.drawArc(arcBounds, start, 255f, false, arcPaint);
            postInvalidateOnAnimation();
        }
    }

    private void installDebugHud(FrameLayout renderFrame) {
        if (debugHudPanel != null || renderFrame == null) {
            return;
        }
        debugHudPanel = new LinearLayout(this);
        debugHudPanel.setOrientation(LinearLayout.VERTICAL);
        debugHudPanel.setPadding(dp(6), dp(3), dp(6), dp(4));
        debugHudPanel.setBackgroundColor(0x99000000);
        debugHudPanel.setClickable(true);
        debugHudPanel.setFocusable(false);
        debugHudPanel.setElevation(dp(48));
        debugHudPanel.setContentDescription(getString(R.string.hud_collapse));
        debugHudPanel.setOnClickListener(view -> {
            view.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
            setDebugHudCollapsed(true);
        });

        frameCaughtCheck = new CheckBox(this);
        frameCaughtCheck.setText(R.string.hud_frame_waiting);
        frameCaughtCheck.setTextColor(0xfff4f7fa);
        frameCaughtCheck.setTextSize(9);
        frameCaughtCheck.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        frameCaughtCheck.setButtonDrawable(
                SettingsActivity.createCheckBoxButtonDrawable(this));
        frameCaughtCheck.setCompoundDrawablePadding(dp(4));
        frameCaughtCheck.setPadding(0, 0, 0, 0);
        frameCaughtCheck.setClickable(false);
        frameCaughtCheck.setFocusable(false);
        frameCaughtCheck.setChecked(false);
        debugHudPanel.addView(frameCaughtCheck, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        debugHud = new TextView(this);
        debugHud.setTextColor(0xeeffffff);
        debugHud.setTextSize(9);
        debugHud.setTypeface(Typeface.MONOSPACE);
        debugHud.setGravity(Gravity.START);
        debugHud.setPadding(0, 0, 0, 0);
        debugHud.setText(getString(R.string.hud_stats_pending));
        debugHud.setClickable(false);
        debugHud.setFocusable(false);
        debugHudPanel.addView(debugHud, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP | Gravity.START);
        params.leftMargin = dp(10);
        params.topMargin = dp(10);
        renderFrame.addView(debugHudPanel, params);

        debugHudCollapsedButton = new TextView(this);
        debugHudCollapsedButton.setText(R.string.hud_metrics);
        debugHudCollapsedButton.setTextColor(0xfff4f7fa);
        debugHudCollapsedButton.setTextSize(7);
        debugHudCollapsedButton.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        debugHudCollapsedButton.setGravity(Gravity.CENTER);
        debugHudCollapsedButton.setIncludeFontPadding(false);
        debugHudCollapsedButton.setSingleLine(true);
        debugHudCollapsedButton.setClickable(true);
        debugHudCollapsedButton.setFocusable(false);
        debugHudCollapsedButton.setElevation(dp(48));
        debugHudCollapsedButton.setContentDescription(getString(R.string.hud_expand));
        GradientDrawable collapsedBackground = new GradientDrawable();
        collapsedBackground.setShape(GradientDrawable.OVAL);
        collapsedBackground.setColor(0xb3000000);
        collapsedBackground.setStroke(dp(1), 0x99ffffff);
        debugHudCollapsedButton.setBackground(collapsedBackground);
        debugHudCollapsedButton.setOnClickListener(view -> {
            view.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
            setDebugHudCollapsed(false);
        });

        FrameLayout.LayoutParams collapsedParams = new FrameLayout.LayoutParams(
                dp(40), dp(40), Gravity.TOP | Gravity.START);
        collapsedParams.leftMargin = dp(10);
        collapsedParams.topMargin = dp(10);
        renderFrame.addView(debugHudCollapsedButton, collapsedParams);
        debugHudCollapsedButton.setVisibility(View.GONE);

        debugHudPanel.bringToFront();
        startRuntimeDebugHud();
    }

    private void setDebugHudCollapsed(boolean collapsed) {
        if (debugHudPanel == null || debugHudCollapsedButton == null) {
            return;
        }
        debugHudPanel.setVisibility(collapsed ? View.GONE : View.VISIBLE);
        debugHudCollapsedButton.setVisibility(collapsed ? View.VISIBLE : View.GONE);
        Log.i(TAG, "EXECUTOR_DEBUG_HUD collapsed=" + collapsed);
        if (collapsed) {
            debugHudCollapsedButton.bringToFront();
        } else {
            debugHudPanel.bringToFront();
            // Sampling continues while collapsed, but the hidden TextView isn't formatted or
            // invalidated once per second. Render the latest counters once when it becomes visible.
            renderRuntimeDebugHud();
        }
    }

    private void applyDebugHudPreference(SharedPreferences preferences) {
        boolean enabled = getIntent().getBooleanExtra("input_debug_hud", false)
                || preferences.getBoolean(SettingsActivity.K_PERF_HUD, false);
        inputDebugHud = enabled;
        if (enabled) {
            if (fullscreenRender && debugHudPanel == null && gameRenderFrame != null) {
                installDebugHud(gameRenderFrame);
                Log.i(TAG, "EXECUTOR_DEBUG_HUD visible=true reason=preference");
            }
            return;
        }
        if (debugHudPanel != null) {
            ViewGroup parent = (ViewGroup) debugHudPanel.getParent();
            if (parent != null) {
                parent.removeView(debugHudPanel);
            }
            debugHudPanel = null;
            debugHud = null;
            frameCaughtCheck = null;
        }
        if (debugHudCollapsedButton != null) {
            ViewGroup parent = (ViewGroup) debugHudCollapsedButton.getParent();
            if (parent != null) {
                parent.removeView(debugHudCollapsedButton);
            }
            debugHudCollapsedButton = null;
        }
        Log.i(TAG, "EXECUTOR_DEBUG_HUD visible=false reason=preference");
    }

    /**
     * Apply a preference to the active render session. Changes that alter Android-owned state are
     * handled on the UI thread; native-facing files are republished immediately and consumed at
     * their existing safe boundary instead of restarting the process.
     */
    private void applyLiveSetting(SharedPreferences preferences, String key) {
        if (launcherMode || preferences == null) {
            return;
        }
        if (key == null || SettingsActivity.K_OVERLAY.equals(key)) {
            overlayEnabledByPreference =
                    preferences.getBoolean(SettingsActivity.K_OVERLAY, true);
            updateTouchOverlayVisibility();
        }
        if (key == null || SettingsActivity.K_OVERLAY_OPACITY.equals(key)) {
            if (inputOverlay != null) {
                int opacity = Math.max(20, Math.min(100,
                        preferences.getInt(SettingsActivity.K_OVERLAY_OPACITY, 74)));
                inputOverlay.setAlpha((opacity / 100f) * DUALSHOCK_ARTWORK_ALPHA);
            }
        }
        if (key == null || SettingsActivity.K_PERF_HUD.equals(key)) {
            applyDebugHudPreference(preferences);
        }
        if (key == null || SettingsActivity.K_KEEP_AWAKE.equals(key)) {
            if (preferences.getBoolean(SettingsActivity.K_KEEP_AWAKE, true)) {
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            } else {
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            }
        }
        if ((key == null || SettingsActivity.K_AUDIO.equals(key)) && runtimeLoaded) {
            int result = RuntimeBridge.setAudioEnabled(
                    preferences.getBoolean(SettingsActivity.K_AUDIO, true));
            Log.i(TAG, "EXECUTOR_LIVE_SETTING audio result=" + result);
        }
        if (key == null || SettingsActivity.K_RES_MODE.equals(key)) {
            File root = new File(getFilesDir(), "lsx4-home");
            materializeRenderResolutionSetting(root);
            if (fullscreenRender && renderSurfaceView != null) {
                int mode = preferences.getInt(
                        SettingsActivity.K_RES_MODE, SettingsActivity.RES_NATIVE);
                int width = SettingsActivity.resolutionWidth(mode);
                int height = SettingsActivity.resolutionHeight(mode);
                if (width > 0 && height > 0) {
                    renderSurfaceView.getHolder().setFixedSize(width, height);
                } else {
                    renderSurfaceView.getHolder().setSizeFromLayout();
                }
                renderSurfaceView.requestLayout();
            }
        }
        if (key == null || SettingsActivity.K_GPU_BACKEND.equals(key)) {
            try {
                GraphicsBackend.selected(this).writeLaunchConfiguration(this);
            } catch (Exception error) {
                Log.e(TAG, "Live graphics backend configuration could not be written", error);
            }
        }
        if (key == null || SettingsActivity.K_PERSISTENT_JIT_CACHE.equals(key)) {
            try {
                materializeJitPersistentJitCacheSetting(
                        new File(getFilesDir(), "lsx4-home"));
            } catch (Exception error) {
                Log.e(TAG, "Live cache setting could not be published", error);
            }
        }
        if (key == null || isGraphicsPatchPreference(key)) {
            materializeGraphicsEffectSettings(new File(getFilesDir(), "lsx4-home"));
        }
    }

    private static boolean isGraphicsPatchPreference(String key) {
        return SettingsActivity.K_DISABLE_DYNAMIC_SHADOWS.equals(key)
                || SettingsActivity.K_DISABLE_SSAO.equals(key)
                || SettingsActivity.K_DISABLE_MOTION_BLUR.equals(key)
                || SettingsActivity.K_DISABLE_DEPTH_OF_FIELD.equals(key)
                || SettingsActivity.K_DISABLE_ANTI_ALIASING.equals(key)
                || SettingsActivity.K_DISABLE_CHROMATIC_ABERRATION.equals(key);
    }

    private void setDebugHud(String text) {
        runtimeHudEvent = text == null || text.isEmpty() ? "-" : text;
        runOnUiThread(() -> {
            if (debugHud != null) {
                renderRuntimeDebugHud();
            }
        });
    }

    private void startRuntimeDebugHud() {
        if (runtimeHudRunning) {
            return;
        }
        runtimeHudRunning = true;
        runtimeHudHandler.post(runtimeHudTick);
    }

    private void resetRuntimeDebugHudForGuestLaunch() {
        Runnable reset = () -> {
            runtimeHudStartedAtMs = SystemClock.elapsedRealtime();
            runtimeHudPreviousSampleAtMs = 0;
            runtimeHudLastSampleAtMs = 0;
            runtimeHudLastStats = null;
            runtimeHudCurrentStats = null;
            frameDetectionInFlight = false;
            frameCaught = false;
            frameCaughtAtMs = 0;
            frameCaughtPath = "";
            frameProbeNonblack = 0;
            frameProbeEdges = 0;
            frameProbeAcceptedStreak = 0;
            if (frameCaughtCheck != null) {
                frameCaughtCheck.setChecked(false);
                frameCaughtCheck.setText(R.string.hud_frame_waiting);
            }
            if (loadingProgressPanel != null) {
                loadingProgressPanel.setVisibility(View.VISIBLE);
            }
            if (loadingBlocksText != null) {
                loadingBlocksText.setText("blocks 0");
            }
            renderRuntimeDebugHud();
        };
        if (Looper.myLooper() == Looper.getMainLooper()) {
            reset.run();
        } else {
            runOnUiThread(reset);
        }
    }

    private void sampleRuntimeHudStats() {
        if (!runtimeLoaded) {
            return;
        }
        try {
            long[] stats = RuntimeBridge.runtimeHudStats();
            if (stats != null && stats.length >= 11) {
                runtimeHudLastStats = runtimeHudCurrentStats;
                runtimeHudCurrentStats = stats;
                runtimeHudPreviousSampleAtMs = runtimeHudLastSampleAtMs;
                runtimeHudLastSampleAtMs = SystemClock.elapsedRealtime();
            }
        } catch (Throwable t) {
            runtimeHudEvent = "HUD stats unavailable: " + t.getClass().getSimpleName();
        }
    }

    private void renderLoadingProgress() {
        if (loadingProgressPanel == null || loadingBlocksText == null) {
            return;
        }
        if (frameCaught) {
            loadingProgressPanel.setVisibility(View.GONE);
            return;
        }
        loadingProgressPanel.setVisibility(View.VISIBLE);
        long blocks = runtimeHudCurrentStats != null && runtimeHudCurrentStats.length > 0
                ? runtimeHudCurrentStats[0] : 0;
        loadingBlocksText.setText(String.format(Locale.US, "blocks %,d", blocks));
    }

    private void renderRuntimeDebugHud() {
        if (debugHud == null || debugHudPanel == null ||
                debugHudPanel.getVisibility() != View.VISIBLE) {
            return;
        }
        long now = SystemClock.elapsedRealtime();
        long elapsedMs = Math.max(0, now - runtimeHudStartedAtMs);
        long totalSeconds = elapsedMs / 1000;
        long minutes = totalSeconds / 60;
        long seconds = totalSeconds % 60;

        long[] current = runtimeHudCurrentStats;
        long[] previous = runtimeHudLastStats;
        if (current == null || current.length < 11) {
            debugHud.setText(String.format(Locale.US,
                    "FPS -- | %02d:%02d\n%s\n%s",
                    minutes, seconds, getString(R.string.hud_runtime_waiting), runtimeHudEvent));
            return;
        }

        long generatedRate = counterRate(current, previous, 0);
        long executionRate = counterRate(current, previous, 4);
        long drawRate = counterRate(current, previous, 6);
        long submitRate = counterRate(current, previous, 8);
        long presentRate = counterRate(current, previous, 10);
        debugHud.setText(String.format(Locale.US,
                "FPS %,d | %02d:%02d\n" +
                        "blocks %,d  +%,d/s\n" +
                        "exec %,d/s | draw %,d/s\n" +
                        "submit %,d/s | present %,d/s\n%s",
                presentRate, minutes, seconds,
                current[0], generatedRate,
                executionRate, drawRate,
                submitRate, presentRate, runtimeHudEvent));
    }

    private String describePersistentCache(long[] stats) {
        if (stats == null || stats.length < 18) {
            return getString(R.string.hud_cache_waiting);
        }
        if (stats[11] == 0) {
            return getString(R.string.hud_cache_disabled);
        }
        boolean hasOldCache = stats[12] > 0 || stats[13] > 0 || stats[14] > 0;
        boolean hasNewCache = stats[15] > 0 || stats[16] > 0 || stats[17] > 0;
        if (hasOldCache && hasNewCache) {
            return getString(R.string.hud_cache_old_and_new);
        }
        if (hasOldCache) {
            return getString(R.string.hud_cache_old);
        }
        return getString(R.string.hud_cache_new);
    }

    private String describePersistentCacheCounters(long[] stats) {
        if (stats == null || stats.length < 18) {
            return "cache counters: runtime " + (stats == null ? 0 : stats.length) + "/18";
        }
        return String.format(Locale.US,
                "cache loaded %,d | IR/native hits %,d/%,d | new cap/IR/native %,d/%,d/%,d",
                stats[12], stats[13], stats[14], stats[15], stats[16], stats[17]);
    }

    private long counterRate(long[] current, long[] previous, int index) {
        if (current == null || previous == null || index >= current.length ||
                index >= previous.length || runtimeHudPreviousSampleAtMs == 0 ||
                runtimeHudLastSampleAtMs <= runtimeHudPreviousSampleAtMs) {
            return 0;
        }
        long delta = Math.max(0, current[index] - previous[index]);
        long sampleMs = runtimeHudLastSampleAtMs - runtimeHudPreviousSampleAtMs;
        return Math.round((double) delta * 1000.0 / (double) sampleMs);
    }

    private void applySoftwareLandscapeIfNeeded(FrameLayout host, View content) {
        int width = host.getWidth();
        int height = host.getHeight();
        if (!fullscreenRender || width <= 0 || height <= 0) {
            return;
        }

        FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) content.getLayoutParams();
        if (height > width) {
            boolean changed = params.width != height || params.height != width ||
                    params.leftMargin != 0 || params.topMargin != 0 ||
                    content.getRotation() != 90.0f ||
                    content.getTranslationX() != width;
            if (changed) {
                params.width = height;
                params.height = width;
                params.leftMargin = 0;
                params.topMargin = 0;
                content.setLayoutParams(params);
                content.setPivotX(0.0f);
                content.setPivotY(0.0f);
                content.setRotation(90.0f);
                content.setTranslationX(width);
                content.setTranslationY(0.0f);
                append("Software landscape applied: host=" + width + "x" + height +
                        " content=" + height + "x" + width);
            }
        } else {
            boolean changed = params.width != ViewGroup.LayoutParams.MATCH_PARENT ||
                    params.height != ViewGroup.LayoutParams.MATCH_PARENT ||
                    params.leftMargin != 0 || params.topMargin != 0 ||
                    content.getRotation() != 0.0f || content.getTranslationX() != 0.0f ||
                    content.getTranslationY() != 0.0f;
            if (changed) {
                params.width = ViewGroup.LayoutParams.MATCH_PARENT;
                params.height = ViewGroup.LayoutParams.MATCH_PARENT;
                params.leftMargin = 0;
                params.topMargin = 0;
                content.setLayoutParams(params);
                content.setPivotX(0.0f);
                content.setPivotY(0.0f);
                content.setRotation(0.0f);
                content.setTranslationX(0.0f);
                content.setTranslationY(0.0f);
                append("Native landscape active: host=" + width + "x" + height);
            }
        }
    }

    private TextView makePadButton(String label, int mask) {
        TextView button = new TextView(this);
        final int[] inputGeneration = {0};
        button.setText(label);
        button.setGravity(Gravity.CENTER);
        button.setTextColor(0xeeffffff);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setTextSize(label.length() > 3 ? 11 : 15);
        button.setBackground(makeOverlayBackground(false));
        button.setAlpha(0.74f);
        button.setFocusable(false);
        button.setClickable(true);
        button.setOnTouchListener((view, event) -> {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                case MotionEvent.ACTION_POINTER_DOWN:
                    inputGeneration[0]++;
                    view.setAlpha(0.96f);
                    button.setBackground(makeOverlayBackground(true));
                    sendPadButton(mask, true);
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_POINTER_UP:
                case MotionEvent.ACTION_CANCEL:
                    view.setAlpha(0.74f);
                    button.setBackground(makeOverlayBackground(false));
                    final int releaseGeneration = ++inputGeneration[0];
                    view.postDelayed(() -> {
                        if (inputGeneration[0] == releaseGeneration) {
                            sendPadButton(mask, false);
                        }
                    }, DIGITAL_BUTTON_RELEASE_LATCH_MS);
                    return true;
                default:
                    return true;
            }
        });
        return button;
    }

    private TextView makeTouchPadButton() {
        TextView button = makePadButton("TOUCH", PAD_TOUCHPAD);
        button.setOnTouchListener((view, event) -> {
            float x = view.getWidth() > 0 ? event.getX() / (float) view.getWidth() : 0.5f;
            float y = view.getHeight() > 0 ? event.getY() / (float) view.getHeight() : 0.5f;
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                case MotionEvent.ACTION_POINTER_DOWN:
                    view.setAlpha(0.96f);
                    button.setBackground(makeOverlayBackground(true));
                    sendTouchPad(true, x, y);
                    return true;
                case MotionEvent.ACTION_MOVE:
                    sendTouchPad(true, x, y);
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_POINTER_UP:
                case MotionEvent.ACTION_CANCEL:
                    view.setAlpha(0.74f);
                    button.setBackground(makeOverlayBackground(false));
                    sendTouchPad(false, x, y);
                    return true;
                default:
                    return true;
            }
        });
        return button;
    }

    private void placeOverlayButton(FrameLayout parent, View view, int left, int top, int size) {
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(size, size);
        params.leftMargin = left;
        params.topMargin = top;
        parent.addView(view, params);
    }

    private void addAnchoredButton(FrameLayout parent, View view, int width, int height, int gravity,
                                   int left, int top, int right, int bottom) {
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(width, height, gravity);
        params.leftMargin = left;
        params.topMargin = top;
        params.rightMargin = right;
        params.bottomMargin = bottom;
        parent.addView(view, params);
    }

    private GradientDrawable makeOverlayBackground(boolean pressed) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(pressed ? 0xaa3b82f6 : 0x66232b35);
        drawable.setStroke(dp(1), pressed ? 0xffffffff : 0xaaffffff);
        drawable.setCornerRadius(dp(12));
        return drawable;
    }

    private final class DualShockOverlayView extends View {
        private static final int AXIS_LEFT_X = 0;
        private static final int AXIS_LEFT_Y = 1;
        private static final int AXIS_RIGHT_X = 2;
        private static final int AXIS_RIGHT_Y = 3;
        private static final int AXIS_L2 = 4;
        private static final int AXIS_R2 = 5;

        private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint iconPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint shadowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Path path = new Path();
        private final RectF body = new RectF();
        private final RectF touchpad = new RectF();
        private final RectF touchpadToggle = new RectF();
        private final RectF share = new RectF();
        private final RectF options = new RectF();
        private final RectF ps = new RectF();
        private final RectF dpadBase = new RectF();
        private final RectF dpadUp = new RectF();
        private final RectF dpadDown = new RectF();
        private final RectF dpadLeft = new RectF();
        private final RectF dpadRight = new RectF();
        private final RectF faceBase = new RectF();
        private final RectF triangle = new RectF();
        private final RectF circle = new RectF();
        private final RectF cross = new RectF();
        private final RectF square = new RectF();
        private final RectF l1 = new RectF();
        private final RectF l2 = new RectF();
        private final RectF r1 = new RectF();
        private final RectF r2 = new RectF();
        private final RectF leftStick = new RectF();
        private final RectF rightStick = new RectF();
        private final RectF scratch = new RectF();
        private final int[] buttonBits = {
                PAD_L3, PAD_R3, PAD_OPTIONS, PAD_UP, PAD_RIGHT, PAD_DOWN, PAD_LEFT,
                PAD_L2, PAD_R2, PAD_L1, PAD_R1, PAD_TRIANGLE, PAD_CIRCLE, PAD_CROSS,
                PAD_SQUARE, PAD_TOUCHPAD
        };

        private float leftStickCenterX;
        private float leftStickCenterY;
        private float rightStickCenterX;
        private float rightStickCenterY;
        private float stickRadius;
        private float stickThrowRadius;
        private int activeMask;
        private int leftX = 128;
        private int leftY = 128;
        private int rightX = 128;
        private int rightY = 128;
        private int triggerLeft;
        private int triggerRight;
        private int leftStickPointerId = -1;
        private int rightStickPointerId = -1;
        private int touchpadPointerId = -1;
        private boolean touchpadDown;
        private boolean touchpadExpanded;
        private float touchpadNormX = 0.5f;
        private float touchpadNormY = 0.5f;
        private boolean geometryDirty = true;
        private boolean motionRefreshPosted;
        private long lastVisualRefreshAtMs;
        private final Runnable delayedMotionRefresh = () -> {
            motionRefreshPosted = false;
            lastVisualRefreshAtMs = SystemClock.uptimeMillis();
            invalidate();
        };

        DualShockOverlayView(Context context) {
            super(context);
            setClickable(true);
            setFocusable(false);
            setWillNotDraw(false);

            fillPaint.setStyle(Paint.Style.FILL);
            strokePaint.setStyle(Paint.Style.STROKE);
            strokePaint.setStrokeCap(Paint.Cap.ROUND);
            strokePaint.setStrokeJoin(Paint.Join.ROUND);
            textPaint.setTextAlign(Paint.Align.CENTER);
            textPaint.setTypeface(Typeface.DEFAULT_BOLD);
            iconPaint.setStyle(Paint.Style.STROKE);
            iconPaint.setStrokeCap(Paint.Cap.ROUND);
            iconPaint.setStrokeJoin(Paint.Join.ROUND);
            shadowPaint.setStyle(Paint.Style.FILL);
            shadowPaint.setColor(0x66000000);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            updateGeometryIfNeeded();

            drawBody(canvas);
            drawShoulders(canvas);
            drawTouchpad(canvas);
            drawCenterButton(canvas, share, "SHARE", false);
            drawCenterButton(canvas, options, "OPTIONS", (activeMask & PAD_OPTIONS) != 0);
            drawPs(canvas);
            drawDpad(canvas);
            drawFaceButtons(canvas);
            drawStick(canvas, leftStick, leftX, leftY, (activeMask & PAD_L3) != 0, "L3");
            drawStick(canvas, rightStick, rightX, rightY, (activeMask & PAD_R3) != 0, "R3");
        }

        @Override
        public boolean onTouchEvent(MotionEvent event) {
            updateGeometryIfNeeded();
            int action = event.getActionMasked();
            if (action == MotionEvent.ACTION_DOWN) {
                float x = event.getX(event.getActionIndex());
                float y = event.getY(event.getActionIndex());
                if (shouldToggleTouchpad(x, y)) {
                    toggleTouchpad();
                    return true;
                }
                if (!isControlHit(x, y)) {
                    return false;
                }
            }
            if (action == MotionEvent.ACTION_CANCEL || action == MotionEvent.ACTION_UP) {
                resetAllInputs();
                return true;
            }

            int releasedPointerIndex = -1;
            if (action == MotionEvent.ACTION_POINTER_UP) {
                releasedPointerIndex = event.getActionIndex();
                int pointerId = event.getPointerId(releasedPointerIndex);
                if (pointerId == leftStickPointerId) {
                    leftStickPointerId = -1;
                }
                if (pointerId == rightStickPointerId) {
                    rightStickPointerId = -1;
                }
                if (pointerId == touchpadPointerId) {
                    touchpadPointerId = -1;
                    if (touchpadDown) {
                        sendTouchPad(false, touchpadNormX, touchpadNormY);
                        touchpadDown = false;
                    }
                }
            }

            if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_POINTER_DOWN) {
                int pointerIndex = event.getActionIndex();
                int pointerId = event.getPointerId(pointerIndex);
                float x = event.getX(pointerIndex);
                float y = event.getY(pointerIndex);
                if (action == MotionEvent.ACTION_POINTER_DOWN && shouldToggleTouchpad(x, y)) {
                    return true;
                }
                if (action == MotionEvent.ACTION_POINTER_DOWN && !isControlHit(x, y)) {
                    return true;
                }
                if (leftStickPointerId == -1 && isInsideStick(x, y, leftStickCenterX, leftStickCenterY)) {
                    leftStickPointerId = pointerId;
                } else if (rightStickPointerId == -1 &&
                        isInsideStick(x, y, rightStickCenterX, rightStickCenterY)) {
                    rightStickPointerId = pointerId;
                } else if (touchpadPointerId == -1 && touchpad.contains(x, y)) {
                    touchpadPointerId = pointerId;
                    updateTouchpad(x, y, true);
                }
            }

            int newMask = 0;
            int newLeftX = 128;
            int newLeftY = 128;
            int newRightX = 128;
            int newRightY = 128;
            boolean sawTouchpad = false;

            for (int i = 0; i < event.getPointerCount(); i++) {
                if (i == releasedPointerIndex) {
                    continue;
                }

                int pointerId = event.getPointerId(i);
                float x = event.getX(i);
                float y = event.getY(i);
                if (pointerId == leftStickPointerId) {
                    newLeftX = axisValue(x, leftStickCenterX);
                    newLeftY = axisValue(y, leftStickCenterY);
                    if (distance(x, y, leftStickCenterX, leftStickCenterY) < stickRadius * 0.38f) {
                        newMask |= PAD_L3;
                    }
                    continue;
                }
                if (pointerId == rightStickPointerId) {
                    newRightX = axisValue(x, rightStickCenterX);
                    newRightY = axisValue(y, rightStickCenterY);
                    if (distance(x, y, rightStickCenterX, rightStickCenterY) < stickRadius * 0.38f) {
                        newMask |= PAD_R3;
                    }
                    continue;
                }
                if (pointerId == touchpadPointerId) {
                    updateTouchpad(x, y, true);
                    newMask |= PAD_TOUCHPAD;
                    sawTouchpad = true;
                    continue;
                }

                int hit = hitMask(x, y);
                newMask |= hit;
                if ((hit & PAD_TOUCHPAD) != 0) {
                    sawTouchpad = true;
                }
            }

            if (!sawTouchpad && touchpadDown && touchpadPointerId == -1) {
                sendTouchPad(false, touchpadNormX, touchpadNormY);
                touchpadDown = false;
            }

            int previousMask = activeMask;
            int previousLeftX = leftX;
            int previousLeftY = leftY;
            int previousRightX = rightX;
            int previousRightY = rightY;
            applyMask(newMask);
            applyAxes(newLeftX, newLeftY, newRightX, newRightY);
            boolean buttonVisualChanged = previousMask != activeMask;
            boolean axisVisualChanged = previousLeftX != leftX || previousLeftY != leftY ||
                    previousRightX != rightX || previousRightY != rightY;
            if (buttonVisualChanged) {
                requestVisualRefresh(true);
            } else if (axisVisualChanged) {
                requestVisualRefresh(false);
            }
            return true;
        }

        @Override
        protected void onSizeChanged(int width, int height, int oldWidth, int oldHeight) {
            super.onSizeChanged(width, height, oldWidth, oldHeight);
            geometryDirty = true;
        }

        @Override
        protected void onDetachedFromWindow() {
            removeCallbacks(delayedMotionRefresh);
            motionRefreshPosted = false;
            lastVisualRefreshAtMs = 0;
            super.onDetachedFromWindow();
        }

        private void updateGeometryIfNeeded() {
            if (!geometryDirty) {
                return;
            }
            updateGeometry();
            geometryDirty = false;
        }

        private void requestVisualRefresh(boolean immediate) {
            long now = SystemClock.uptimeMillis();
            long elapsed = now - lastVisualRefreshAtMs;
            if (immediate || lastVisualRefreshAtMs == 0 ||
                    elapsed >= OVERLAY_MOTION_REDRAW_INTERVAL_MS) {
                if (motionRefreshPosted) {
                    removeCallbacks(delayedMotionRefresh);
                    motionRefreshPosted = false;
                }
                lastVisualRefreshAtMs = now;
                invalidate();
                return;
            }
            if (!motionRefreshPosted) {
                motionRefreshPosted = true;
                postDelayed(delayedMotionRefresh,
                        Math.max(1L, OVERLAY_MOTION_REDRAW_INTERVAL_MS - elapsed));
            }
        }

        private void updateGeometry() {
            float width = Math.max(1f, getWidth());
            float height = Math.max(1f, getHeight());
            float min = Math.min(width, height);
            float sideMargin = Math.max(dp(8), width * 0.018f);
            float bottomInset = Math.max(dp(28), height * 0.035f);
            float bottom = height - bottomInset;

            body.set(width * 0.33f, height * 0.940f, width * 0.67f, height + dp(10));
            float psSize = Math.max(dp(28), Math.min(min * 0.042f, dp(44)));
            float psCenterY = bottom - Math.max(dp(16), min * 0.020f);
            setCentered(ps, width * 0.5f, psCenterY, psSize, psSize);

            float touchW = touchpadExpanded
                    ? Math.min(width * 0.34f, min * 0.58f)
                    : Math.min(width * 0.18f, min * 0.32f);
            float touchH = touchpadExpanded
                    ? Math.max(dp(76), min * 0.130f)
                    : Math.max(dp(28), min * 0.040f);
            float touchGap = touchpadExpanded ? Math.max(dp(12), min * 0.020f)
                    : Math.max(dp(10), min * 0.016f);
            float touchTop = ps.top - touchGap - touchH;
            touchpad.set(width * 0.5f - touchW * 0.5f, touchTop,
                    width * 0.5f + touchW * 0.5f, touchTop + touchH);
            if (touchpadExpanded) {
                float toggleW = Math.max(dp(54), touchpad.width() * 0.18f);
                float toggleH = Math.max(dp(16), min * 0.023f);
                setCentered(touchpadToggle, touchpad.centerX(), touchpad.top + toggleH * 0.80f,
                        toggleW, toggleH);
            } else {
                touchpadToggle.set(touchpad);
            }
            float centerW = Math.max(dp(48), Math.min(min * 0.076f, dp(78)));
            float centerH = Math.max(dp(22), Math.min(min * 0.038f, dp(34)));
            float centerGap = Math.max(dp(12), min * 0.020f);
            float centerY = touchpad.centerY();
            share.set(touchpad.left - centerW - centerGap, centerY - centerH * 0.5f,
                    touchpad.left - centerGap, centerY + centerH * 0.5f);
            options.set(touchpad.right + centerGap, centerY - centerH * 0.5f,
                    touchpad.right + centerW + centerGap, centerY + centerH * 0.5f);

            float cluster = Math.max(dp(112), Math.min(min * 0.285f, dp(156)));
            float unit = cluster * 0.325f;
            float padSize = cluster * 0.350f;
            float dpadX = sideMargin + cluster * 0.46f;
            float dpadY = bottom - cluster * 0.540f;
            setCentered(dpadBase, dpadX, dpadY, cluster, cluster);
            setCentered(dpadUp, dpadX, dpadY - unit, padSize, padSize);
            setCentered(dpadDown, dpadX, dpadY + unit, padSize, padSize);
            setCentered(dpadLeft, dpadX - unit, dpadY, padSize, padSize);
            setCentered(dpadRight, dpadX + unit, dpadY, padSize, padSize);

            float faceX = width - sideMargin - cluster * 0.46f;
            float faceY = dpadY;
            setCentered(faceBase, faceX, faceY, cluster, cluster);
            float faceSize = cluster * 0.385f;
            setCentered(triangle, faceX, faceY - unit, faceSize, faceSize);
            setCentered(cross, faceX, faceY + unit, faceSize, faceSize);
            setCentered(square, faceX - unit, faceY, faceSize, faceSize);
            setCentered(circle, faceX + unit, faceY, faceSize, faceSize);

            float shoulderW = Math.min(width * 0.090f, min * 0.180f);
            float shoulderH = Math.max(dp(18), min * 0.032f);
            float shoulderGap = Math.max(dp(12), width * 0.009f);
            float shoulderTop = Math.max(dp(12), height * 0.078f);
            l1.set(sideMargin, shoulderTop, sideMargin + shoulderW, shoulderTop + shoulderH);
            l2.set(l1.right + shoulderGap, shoulderTop, l1.right + shoulderGap + shoulderW,
                    shoulderTop + shoulderH);
            r2.set(width - sideMargin - shoulderW, shoulderTop, width - sideMargin,
                    shoulderTop + shoulderH);
            r1.set(r2.left - shoulderGap - shoulderW, shoulderTop, r2.left - shoulderGap,
                    shoulderTop + shoulderH);

            stickRadius = Math.max(dp(24), Math.min(min * 0.060f, dp(42)));
            stickThrowRadius = stickRadius * 0.46f;
            leftStickCenterX = Math.max(width * 0.30f, dpadBase.right + stickRadius * 1.15f);
            leftStickCenterY = Math.min(dpadY + cluster * 0.44f,
                    height - bottomInset - stickRadius * 0.14f);
            rightStickCenterX = Math.min(width * 0.70f, faceBase.left - stickRadius * 1.15f);
            rightStickCenterY = leftStickCenterY;
            setCentered(leftStick, leftStickCenterX, leftStickCenterY, stickRadius * 2f,
                    stickRadius * 2f);
            setCentered(rightStick, rightStickCenterX, rightStickCenterY, stickRadius * 2f,
                    stickRadius * 2f);

            float stroke = Math.max(dp(1), min * 0.0035f);
            strokePaint.setStrokeWidth(stroke);
            iconPaint.setStrokeWidth(stroke * 1.25f);
            textPaint.setTextSize(Math.max(dp(8), Math.min(min * 0.013f, dp(12))));
        }

        private void drawBody(Canvas canvas) {
            fillPaint.setColor(0x3f05070b);
            canvas.drawRoundRect(body, body.height() * 0.45f, body.height() * 0.45f, fillPaint);
            strokePaint.setColor(0x565a6572);
            canvas.drawRoundRect(body, body.height() * 0.45f, body.height() * 0.45f, strokePaint);
            float barW = touchpad.width() * 0.62f;
            scratch.set(touchpad.centerX() - barW * 0.5f, touchpad.top - dp(8),
                    touchpad.centerX() + barW * 0.5f, touchpad.top - dp(5));
            fillPaint.setColor(0x904aa3ff);
            canvas.drawRoundRect(scratch, dp(3), dp(3), fillPaint);
        }

        private void drawShoulders(Canvas canvas) {
            drawShoulder(canvas, l1, "L1", (activeMask & PAD_L1) != 0);
            drawShoulder(canvas, l2, "L2", (activeMask & PAD_L2) != 0);
            drawShoulder(canvas, r1, "R1", (activeMask & PAD_R1) != 0);
            drawShoulder(canvas, r2, "R2", (activeMask & PAD_R2) != 0);
        }

        private void drawShoulder(Canvas canvas, RectF rect, String label, boolean pressed) {
            fillPaint.setColor(pressed ? 0xc52a6ee8 : 0x76080b11);
            canvas.drawRoundRect(rect, rect.height() * 0.48f, rect.height() * 0.48f, fillPaint);
            strokePaint.setColor(pressed ? 0xeeffffff : 0x7fa7b0bf);
            canvas.drawRoundRect(rect, rect.height() * 0.48f, rect.height() * 0.48f, strokePaint);
            drawCenteredText(canvas, rect, label, 0xeeffffff);
        }

        private void drawTouchpad(Canvas canvas) {
            boolean pressed = (activeMask & PAD_TOUCHPAD) != 0;
            float corner = touchpadExpanded ? dp(12) : touchpad.height() * 0.48f;
            fillPaint.setColor(pressed ? 0xc4243244 : 0x7410151f);
            canvas.drawRoundRect(touchpad, corner, corner, fillPaint);
            strokePaint.setColor(pressed ? 0xff9ad7ff : 0x86a8b2c2);
            canvas.drawRoundRect(touchpad, corner, corner, strokePaint);
            iconPaint.setColor(pressed ? 0x88d5f1ff : 0x55d5f1ff);
            if (touchpadExpanded) {
                float step = touchpad.width() / 5f;
                for (int i = 1; i < 5; i++) {
                    float x = touchpad.left + step * i;
                    canvas.drawLine(x, touchpad.top + dp(18), x, touchpad.bottom - dp(12), iconPaint);
                }
                drawCenteredText(canvas, touchpad, "TOUCH PAD", 0xcce8eef6);
            } else {
                drawCenteredText(canvas, touchpad, "TOUCH", 0xcce8eef6);
            }
        }

        private void drawCenterButton(Canvas canvas, RectF rect, String label, boolean pressed) {
            fillPaint.setColor(pressed ? 0xd735485f : 0x7f05070a);
            canvas.drawRoundRect(rect, rect.height() * 0.5f, rect.height() * 0.5f, fillPaint);
            strokePaint.setColor(pressed ? 0xffe7eef8 : 0x7fa2aab6);
            canvas.drawRoundRect(rect, rect.height() * 0.5f, rect.height() * 0.5f, strokePaint);
            drawCenteredText(canvas, rect, label, 0xdde9eef6);
        }

        private void drawPs(Canvas canvas) {
            boolean pressed = false;
            float radius = ps.width() * 0.5f;
            fillPaint.setColor(pressed ? 0xdd2d3744 : 0xbb05070a);
            canvas.drawCircle(ps.centerX(), ps.centerY(), radius, fillPaint);
            strokePaint.setColor(0xaadce4ee);
            canvas.drawCircle(ps.centerX(), ps.centerY(), radius, strokePaint);
            drawCenteredText(canvas, ps, "PS", 0xeef0f5ff);
        }

        private void drawDpad(Canvas canvas) {
            drawClusterBase(canvas, dpadBase, (activeMask & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT)) != 0);
            drawDpadButton(canvas, dpadUp, "up", (activeMask & PAD_UP) != 0);
            drawDpadButton(canvas, dpadDown, "down", (activeMask & PAD_DOWN) != 0);
            drawDpadButton(canvas, dpadLeft, "left", (activeMask & PAD_LEFT) != 0);
            drawDpadButton(canvas, dpadRight, "right", (activeMask & PAD_RIGHT) != 0);
        }

        private void drawClusterBase(Canvas canvas, RectF rect, boolean active) {
            fillPaint.setColor(active ? 0x771b2634 : 0x4d05070a);
            canvas.drawCircle(rect.centerX(), rect.centerY(), rect.width() * 0.5f, fillPaint);
            strokePaint.setColor(active ? 0x88d8e4f2 : 0x55414a56);
            canvas.drawCircle(rect.centerX(), rect.centerY(), rect.width() * 0.5f, strokePaint);
        }

        private void drawDpadButton(Canvas canvas, RectF rect, String direction, boolean pressed) {
            float radius = rect.width() * 0.24f;
            fillPaint.setColor(pressed ? 0xdd344052 : 0xe3080a0d);
            canvas.drawRoundRect(rect, radius, radius, fillPaint);
            strokePaint.setColor(pressed ? 0xfff0f5ff : 0xaa6c7480);
            canvas.drawRoundRect(rect, radius, radius, strokePaint);
            float cx = rect.centerX();
            float cy = rect.centerY();
            float s = rect.width() * 0.17f;
            iconPaint.setColor(pressed ? 0xffffffff : 0xccd8dde5);
            if ("up".equals(direction)) {
                canvas.drawLine(cx, cy - s, cx - s, cy + s * 0.65f, iconPaint);
                canvas.drawLine(cx, cy - s, cx + s, cy + s * 0.65f, iconPaint);
            } else if ("down".equals(direction)) {
                canvas.drawLine(cx, cy + s, cx - s, cy - s * 0.65f, iconPaint);
                canvas.drawLine(cx, cy + s, cx + s, cy - s * 0.65f, iconPaint);
            } else if ("left".equals(direction)) {
                canvas.drawLine(cx - s, cy, cx + s * 0.65f, cy - s, iconPaint);
                canvas.drawLine(cx - s, cy, cx + s * 0.65f, cy + s, iconPaint);
            } else {
                canvas.drawLine(cx + s, cy, cx - s * 0.65f, cy - s, iconPaint);
                canvas.drawLine(cx + s, cy, cx - s * 0.65f, cy + s, iconPaint);
            }
        }

        private void drawFaceButtons(Canvas canvas) {
            drawClusterBase(canvas, faceBase,
                    (activeMask & (PAD_TRIANGLE | PAD_CIRCLE | PAD_CROSS | PAD_SQUARE)) != 0);
            drawFace(canvas, triangle, "triangle", 0xff3bd671, (activeMask & PAD_TRIANGLE) != 0);
            drawFace(canvas, circle, "circle", 0xffff5a68, (activeMask & PAD_CIRCLE) != 0);
            drawFace(canvas, cross, "cross", 0xff68a8ff, (activeMask & PAD_CROSS) != 0);
            drawFace(canvas, square, "square", 0xffff7ccc, (activeMask & PAD_SQUARE) != 0);
        }

        private void drawFace(Canvas canvas, RectF rect, String symbol, int color, boolean pressed) {
            float radius = rect.width() * 0.5f;
            fillPaint.setColor(pressed ? 0xee243044 : 0xdd05070a);
            canvas.drawCircle(rect.centerX(), rect.centerY(), radius, fillPaint);
            strokePaint.setColor(pressed ? 0xffffffff : 0xaa89919c);
            canvas.drawCircle(rect.centerX(), rect.centerY(), radius, strokePaint);
            iconPaint.setColor(color);
            float cx = rect.centerX();
            float cy = rect.centerY();
            float s = rect.width() * 0.22f;
            if ("triangle".equals(symbol)) {
                path.reset();
                path.moveTo(cx, cy - s);
                path.lineTo(cx - s * 1.05f, cy + s * 0.85f);
                path.lineTo(cx + s * 1.05f, cy + s * 0.85f);
                path.close();
                canvas.drawPath(path, iconPaint);
            } else if ("circle".equals(symbol)) {
                canvas.drawCircle(cx, cy, s * 0.96f, iconPaint);
            } else if ("cross".equals(symbol)) {
                canvas.drawLine(cx - s, cy - s, cx + s, cy + s, iconPaint);
                canvas.drawLine(cx + s, cy - s, cx - s, cy + s, iconPaint);
            } else {
                scratch.set(cx - s, cy - s, cx + s, cy + s);
                canvas.drawRect(scratch, iconPaint);
            }
        }

        private void drawStick(Canvas canvas, RectF rect, int axisX, int axisY, boolean pressed, String label) {
            float radius = rect.width() * 0.5f;
            fillPaint.setColor(pressed ? 0xcc293748 : 0xba05070a);
            canvas.drawCircle(rect.centerX(), rect.centerY(), radius, fillPaint);
            strokePaint.setColor(pressed ? 0xffe8f1ff : 0xaaaab2bf);
            canvas.drawCircle(rect.centerX(), rect.centerY(), radius, strokePaint);

            float knobX = rect.centerX() + ((axisX - 128) / 127f) * stickThrowRadius;
            float knobY = rect.centerY() + ((axisY - 128) / 127f) * stickThrowRadius;
            fillPaint.setColor(0xee111821);
            canvas.drawCircle(knobX, knobY, radius * 0.46f, fillPaint);
            strokePaint.setColor(0xff5f6977);
            canvas.drawCircle(knobX, knobY, radius * 0.46f, strokePaint);
            scratch.set(knobX - radius * 0.35f, knobY - radius * 0.15f,
                    knobX + radius * 0.35f, knobY + radius * 0.15f);
            drawCenteredText(canvas, scratch, label, 0xbbffffff);
        }

        private int hitMask(float x, float y) {
            return controlMaskAt(x, y, true);
        }

        private int controlMaskAt(float x, float y, boolean includeTouchpad) {
            int mask = 0;
            if (dpadUp.contains(x, y)) mask |= PAD_UP;
            if (dpadRight.contains(x, y)) mask |= PAD_RIGHT;
            if (dpadDown.contains(x, y)) mask |= PAD_DOWN;
            if (dpadLeft.contains(x, y)) mask |= PAD_LEFT;
            if (triangle.contains(x, y)) mask |= PAD_TRIANGLE;
            if (circle.contains(x, y)) mask |= PAD_CIRCLE;
            if (cross.contains(x, y)) mask |= PAD_CROSS;
            if (square.contains(x, y)) mask |= PAD_SQUARE;
            if (l1.contains(x, y)) mask |= PAD_L1;
            if (l2.contains(x, y)) mask |= PAD_L2;
            if (r1.contains(x, y)) mask |= PAD_R1;
            if (r2.contains(x, y)) mask |= PAD_R2;
            if (options.contains(x, y)) mask |= PAD_OPTIONS;
            if (leftStick.contains(x, y)) mask |= PAD_L3;
            if (rightStick.contains(x, y)) mask |= PAD_R3;
            if (includeTouchpad && touchpadExpanded && touchpad.contains(x, y) &&
                    !touchpadToggle.contains(x, y)) {
                mask |= PAD_TOUCHPAD;
            }
            return mask;
        }

        private boolean isControlHit(float x, float y) {
            return controlMaskAt(x, y, true) != 0 ||
                    isInsideStick(x, y, leftStickCenterX, leftStickCenterY) ||
                    isInsideStick(x, y, rightStickCenterX, rightStickCenterY) ||
                    containsWithSlop(touchpad, x, y, dp(12)) ||
                    ps.contains(x, y);
        }

        private boolean shouldToggleTouchpad(float x, float y) {
            float slop = dp(14);
            return containsWithSlop(touchpad, x, y, slop) &&
                    (!touchpadExpanded || containsWithSlop(touchpadToggle, x, y, slop));
        }

        private boolean containsWithSlop(RectF rect, float x, float y, float slop) {
            return x >= rect.left - slop && x <= rect.right + slop &&
                    y >= rect.top - slop && y <= rect.bottom + slop;
        }

        private void toggleTouchpad() {
            if (touchpadDown) {
                sendTouchPad(false, touchpadNormX, touchpadNormY);
                touchpadDown = false;
            }
            touchpadPointerId = -1;
            touchpadExpanded = !touchpadExpanded;
            geometryDirty = true;
            performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
            requestVisualRefresh(true);
        }

        private void applyMask(int newMask) {
            int changed = activeMask ^ newMask;
            if (changed == 0) {
                return;
            }
            for (int bit : buttonBits) {
                if ((changed & bit) != 0) {
                    boolean pressed = (newMask & bit) != 0;
                    if (pressed) {
                        performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
                    }
                    sendPadButton(bit, pressed);
                }
            }
            activeMask = newMask;
        }

        private void applyAxes(int newLeftX, int newLeftY, int newRightX, int newRightY) {
            if (Math.abs(leftX - newLeftX) > 1) {
                leftX = newLeftX;
                sendPadAxis(AXIS_LEFT_X, leftX, false);
            }
            if (Math.abs(leftY - newLeftY) > 1) {
                leftY = newLeftY;
                sendPadAxis(AXIS_LEFT_Y, leftY, false);
            }
            if (Math.abs(rightX - newRightX) > 1) {
                rightX = newRightX;
                sendPadAxis(AXIS_RIGHT_X, rightX, false);
            }
            if (Math.abs(rightY - newRightY) > 1) {
                rightY = newRightY;
                sendPadAxis(AXIS_RIGHT_Y, rightY, false);
            }
            int l2Value = (activeMask & PAD_L2) != 0 ? 255 : 0;
            int r2Value = (activeMask & PAD_R2) != 0 ? 255 : 0;
            if (triggerLeft != l2Value) {
                triggerLeft = l2Value;
                sendPadAxis(AXIS_L2, triggerLeft);
            }
            if (triggerRight != r2Value) {
                triggerRight = r2Value;
                sendPadAxis(AXIS_R2, triggerRight);
            }
        }

        private void updateTouchpad(float x, float y, boolean pressed) {
            touchpadNormX = clamp01((x - touchpad.left) / Math.max(1f, touchpad.width()));
            touchpadNormY = clamp01((y - touchpad.top) / Math.max(1f, touchpad.height()));
            if (!touchpadDown || pressed) {
                sendTouchPad(pressed, touchpadNormX, touchpadNormY);
                touchpadDown = pressed;
            }
        }

        private void resetAllInputs() {
            boolean visualStateChanged = activeMask != 0 || leftX != 128 || leftY != 128 ||
                    rightX != 128 || rightY != 128;
            leftStickPointerId = -1;
            rightStickPointerId = -1;
            touchpadPointerId = -1;
            if (touchpadDown) {
                sendTouchPad(false, touchpadNormX, touchpadNormY);
                touchpadDown = false;
            }
            applyMask(0);
            applyAxes(128, 128, 128, 128);
            if (visualStateChanged) {
                requestVisualRefresh(true);
            }
        }

        private int axisValue(float value, float center) {
            float normalized = Math.max(-1f, Math.min(1f, (value - center) / stickThrowRadius));
            return Math.max(0, Math.min(255, Math.round(128 + normalized * 127f)));
        }

        private boolean isInsideStick(float x, float y, float centerX, float centerY) {
            return distance(x, y, centerX, centerY) <= stickRadius * 1.15f;
        }

        private float distance(float x, float y, float centerX, float centerY) {
            float dx = x - centerX;
            float dy = y - centerY;
            return (float) Math.sqrt(dx * dx + dy * dy);
        }

        private float clamp01(float value) {
            return Math.max(0f, Math.min(1f, value));
        }

        private void drawCenteredText(Canvas canvas, RectF rect, String label, int color) {
            textPaint.setColor(color);
            Paint.FontMetrics metrics = textPaint.getFontMetrics();
            float baseline = rect.centerY() - (metrics.ascent + metrics.descent) * 0.5f;
            canvas.drawText(label, rect.centerX(), baseline, textPaint);
        }

        private void setCentered(RectF rect, float centerX, float centerY, float width, float height) {
            rect.set(centerX - width * 0.5f, centerY - height * 0.5f,
                    centerX + width * 0.5f, centerY + height * 0.5f);
        }
    }

    private boolean handlePadKey(int keyCode, boolean pressed, KeyEvent event) {
        InputDevice device = event == null ? null : event.getDevice();
        boolean controllerEvent = event != null && (isControllerSource(event.getSource())
                || (device != null && isControllerSource(device.getSources())));
        int mask = padMaskForKey(keyCode, controllerEvent, device);
        if (pressed && event != null && event.getRepeatCount() > 0) {
            return controllerEvent || mask != 0;
        }
        if (controllerEvent) {
            String eligibility = gamepadEligibility(device);
            Log.i(TAG, String.format(Locale.US,
                    "EXECUTOR_GAMEPAD_BUTTON deviceId=%d descriptor=%s vid=%04x pid=%04x " +
                            "keyCode=%d keyName=%s scanCode=%d action=%s repeat=%d " +
                            "mappedMask=0x%08x eligibility=%s",
                    device == null ? -1 : device.getId(),
                    device == null ? "" : device.getDescriptor(),
                    device == null ? 0 : device.getVendorId(),
                    device == null ? 0 : device.getProductId(), keyCode,
                    KeyEvent.keyCodeToString(keyCode), event == null ? 0 : event.getScanCode(),
                    pressed ? "DOWN" : "UP", event == null ? 0 : event.getRepeatCount(), mask,
                    eligibility));
            if (!"eligible".equals(eligibility)) {
                // Do not let an OEM software GAMEPAD fall through to the keyboard path and send
                // phantom PS4 buttons (or close the Activity through KEYCODE_BACK).
                return true;
            }
            if (externalGamepadDeviceId >= 0 && device != null
                    && device.getId() != externalGamepadDeviceId) {
                return true;
            }
            noteExternalGamepadEvent(device);
            if (mask == 0) {
                // PS/Mode has no public OrbisPadButtonDataOffset; consume it without inventing a
                // guest bit. Unknown keys remain visible in the diagnostic line above.
                return keyCode == KeyEvent.KEYCODE_BUTTON_MODE
                        || keyCode == KeyEvent.KEYCODE_BUTTON_SELECT
                        || keyCode == KeyEvent.KEYCODE_BUTTON_1;
            }
            if (pressed) {
                externalButtonMask |= mask;
            } else {
                externalButtonMask &= ~mask;
            }
            applyExternalDigitalMask();
            return true;
        }
        if (mask == 0) {
            return false;
        }
        sendPadButton(mask, pressed);
        return true;
    }

    private int padMaskForKey(int keyCode, boolean controllerEvent, InputDevice device) {
        switch (keyCode) {
            case KeyEvent.KEYCODE_DPAD_UP:
                return PAD_UP;
            case KeyEvent.KEYCODE_DPAD_RIGHT:
                return PAD_RIGHT;
            case KeyEvent.KEYCODE_DPAD_DOWN:
                return PAD_DOWN;
            case KeyEvent.KEYCODE_DPAD_LEFT:
                return PAD_LEFT;
            case KeyEvent.KEYCODE_DPAD_UP_LEFT:
                return PAD_UP | PAD_LEFT;
            case KeyEvent.KEYCODE_DPAD_UP_RIGHT:
                return PAD_UP | PAD_RIGHT;
            case KeyEvent.KEYCODE_DPAD_DOWN_LEFT:
                return PAD_DOWN | PAD_LEFT;
            case KeyEvent.KEYCODE_DPAD_DOWN_RIGHT:
                return PAD_DOWN | PAD_RIGHT;
            case KeyEvent.KEYCODE_DPAD_CENTER:
            case KeyEvent.KEYCODE_ENTER:
            case KeyEvent.KEYCODE_SPACE:
            case KeyEvent.KEYCODE_BUTTON_A:
                return PAD_CROSS;
            case KeyEvent.KEYCODE_BACK:
                return controllerEvent ? PAD_TOUCHPAD : PAD_CIRCLE;
            case KeyEvent.KEYCODE_ESCAPE:
            case KeyEvent.KEYCODE_BUTTON_B:
                return PAD_CIRCLE;
            case KeyEvent.KEYCODE_BUTTON_X:
                return PAD_SQUARE;
            case KeyEvent.KEYCODE_BUTTON_Y:
                return PAD_TRIANGLE;
            case KeyEvent.KEYCODE_BUTTON_L1:
                return PAD_L1;
            case KeyEvent.KEYCODE_BUTTON_R1:
                return PAD_R1;
            case KeyEvent.KEYCODE_BUTTON_L2:
                return PAD_L2;
            case KeyEvent.KEYCODE_BUTTON_R2:
                return PAD_R2;
            case KeyEvent.KEYCODE_BUTTON_THUMBL:
                return PAD_L3;
            case KeyEvent.KEYCODE_BUTTON_THUMBR:
                return PAD_R3;
            case KeyEvent.KEYCODE_MENU:
            case KeyEvent.KEYCODE_BUTTON_START:
                return PAD_OPTIONS;
            case KeyEvent.KEYCODE_BUTTON_SELECT:
                // Generic SDL mappings commonly expose BACK/SELECT as touchpad. A Sony-compatible
                // device has a distinct Share button, however, and Share has no OrbisPad bit.
                return controllerEvent && !isSonyCompatibleController(device) ? PAD_TOUCHPAD : 0;
            case KeyEvent.KEYCODE_BUTTON_1:
                // DualSense Android fallback exposes touchpad press as BUTTON_1. Do not select this
                // by VID/PID alone: GameSir G8 spoofs 054c:0ce6 without being a DualSense.
                return controllerEvent && isSonyCompatibleController(device)
                        ? PAD_TOUCHPAD : 0;
            case KeyEvent.KEYCODE_BUTTON_MODE:
                return 0;
            default:
                return 0;
        }
    }

    private boolean isSonyCompatibleController(InputDevice device) {
        if (device == null || device.getVendorId() != 0x054c) {
            return false;
        }
        String name = device.getName() == null ? "" : device.getName().toLowerCase(Locale.ROOT);
        return !name.contains("gamesir") && (name.contains("dualsense")
                || name.contains("dualshock") || name.contains("wireless controller"));
    }

    private int helperPadMask = 0;
    // Live analog state mirrored into the pad-state file (0x80 = centered). The OpenOrbis SDL-PS4
    // sample moves the ship via the LEFT STICK axes, so these MUST flow to the guest, not just
    // buttons. Previously the sticks were hardcoded to 0x80 here, so L3 drags never reached SDL.
    private int helperLeftX = 0x80, helperLeftY = 0x80, helperRightX = 0x80, helperRightY = 0x80;

    /** Write OrbisPadData to the file the helper / :elfbox64 guest's libScePad shim reads, so the
     *  on-screen gamepad (buttons AND analog sticks) reaches the guest. */
    private void updateHelperPadState(int mask, boolean pressed) {
        if (pressed) helperPadMask |= mask; else helperPadMask &= ~mask;
        if (isHelperPadStateRequired()) {
            writeHelperPadState();
        }
    }

    /** Mirror a live analog axis (0..255, 0x80 centered) into the pad-state file. */
    private void updateHelperPadAxis(int axis, int value) {
        int v = Math.max(0, Math.min(255, value));
        switch (axis) {
            case 0: helperLeftX = v; break;   // AXIS_LEFT_X
            case 1: helperLeftY = v; break;   // AXIS_LEFT_Y
            case 2: helperRightX = v; break;  // AXIS_RIGHT_X
            case 3: helperRightY = v; break;  // AXIS_RIGHT_Y
            default: return;                  // L2/R2 triggers not part of OrbisPadData sticks
        }
        if (isHelperPadStateRequired()) {
            writeHelperPadState();
        }
    }

    private boolean isHelperPadStateRequired() {
        Intent intent = getIntent();
        boolean legacyEmbeddedBox64 =
                (mappedEntryRunRequested ||
                        intentFlag(intent, EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY));
        return legacyEmbeddedBox64 ||
                intentFlag(intent, EXTRA_TRANSLATOR_BRIDGE_LAUNCH) ||
                intentFlag(intent, EXTRA_JAVA_TRANSLATOR_SMOKE) ||
                (intent != null &&
                        intent.getStringExtra(EXTRA_TRANSLATOR_GUEST_ELF_PATH) != null);
    }

    private void writeHelperPadState() {
        try {
            byte[] d = new byte[256];
            d[0] = (byte) (helperPadMask & 0xFF);
            d[1] = (byte) ((helperPadMask >> 8) & 0xFF);
            d[2] = (byte) ((helperPadMask >> 16) & 0xFF);
            d[3] = (byte) ((helperPadMask >> 24) & 0xFF);
            d[4] = (byte) helperLeftX;  d[5] = (byte) helperLeftY;   // left stick
            d[6] = (byte) helperRightX; d[7] = (byte) helperRightY;  // right stick
            d[0x4c] = 1; // OpenOrbis OrbisPadData `connected` offset (the one SDL-PS4 checks)
            d[0x64] = 1; // helper-path compat
            java.io.File f = new java.io.File(new java.io.File(getFilesDir(), "lsx4-home/translator"),
                    "executor-pad-state.bin");
            try (java.io.FileOutputStream out = new java.io.FileOutputStream(f)) { out.write(d); }
        } catch (Exception e) { /* helper path optional */ }
    }

    private void sendPadButton(int mask, boolean pressed) {
        sendPadButton(mask, pressed, true);
    }

    private void sendPadButton(int mask, boolean pressed, boolean hostFeedback) {
        try {
            int result = RuntimeBridge.setPadButton(mask, pressed);
            updateHelperPadState(mask, pressed);
            String line = String.format(Locale.US,
                    "Pad input: {\"mask\":\"0x%08x\",\"pressed\":%s,\"result\":%d}",
                    mask, pressed ? "true" : "false", result);
            Log.i(TAG, line);
            appendSmokeLog(line);
            setDebugHud(String.format(Locale.US, "PAD 0x%08x %s result=%d", mask,
                    pressed ? "DOWN" : "UP", result));
            if (pressed) {
                if (hostFeedback) {
                    playUiClickTone(mask);
                }
                handleStorePreviewPad(mask);
            }
        } catch (Throwable t) {
            Log.i(TAG, "Pad input failed: " + t.getMessage());
            setDebugHud("PAD failed: " + t.getMessage());
        }
    }

    private void sendPadAxis(int axis, int value) {
        sendPadAxis(axis, value, true);
    }

    private void sendPadAxis(int axis, int value, boolean verbose) {
        try {
            int clamped = Math.max(0, Math.min(255, value));
            int result = RuntimeBridge.setPadAxis(axis, clamped);
            updateHelperPadAxis(axis, clamped);
            if (verbose) {
                String line = String.format(Locale.US,
                        "Pad axis: {\"axis\":%d,\"value\":%d,\"result\":%d}",
                        axis, clamped, result);
                Log.i(TAG, line);
                appendSmokeLog(line);
            }
        } catch (Throwable t) {
            Log.i(TAG, "Pad axis failed: " + t.getMessage());
            setDebugHud("AXIS failed: " + t.getMessage());
        }
    }

    private void playUiClickTone(int mask) {
        try {
            if (uiTone == null) {
                uiTone = new ToneGenerator(AudioManager.STREAM_MUSIC, 88);
            }
            int tone = mask == PAD_CROSS ? ToneGenerator.TONE_PROP_ACK :
                    mask == PAD_CIRCLE ? ToneGenerator.TONE_PROP_NACK :
                            ToneGenerator.TONE_PROP_BEEP;
            uiTone.startTone(tone, mask == PAD_CROSS ? 90 : 45);
        } catch (Throwable t) {
            appendSmokeLog("UI click tone failed: " + t.getMessage());
        }
    }

    private void handleStorePreviewPad(int mask) {
        StoreCatalogView preview = storeCatalogPreview;
        if (preview == null) {
            return;
        }
        Runnable update = () -> {
            String status = preview.handlePad(mask);
            if (status != null && !status.isEmpty()) {
                appendSmokeLog(String.format(Locale.US,
                        "Store catalog preview input: {\"mask\":\"0x%08x\",\"status\":\"%s\"}",
                        mask, sanitizeJson(status)));
                setDebugHud(status);
            }
        };
        if (Looper.myLooper() == Looper.getMainLooper()) {
            update.run();
        } else {
            runOnUiThread(update);
        }
    }

    private void sendTouchPad(boolean pressed, float x, float y) {
        try {
            int result = RuntimeBridge.setTouchPad(pressed,
                    Math.max(0.0f, Math.min(1.0f, x)),
                    Math.max(0.0f, Math.min(1.0f, y)));
            String line = String.format(Locale.US,
                    "Touchpad input: {\"pressed\":%s,\"x\":%.3f,\"y\":%.3f,\"result\":%d}",
                    pressed ? "true" : "false", x, y, result);
            Log.i(TAG, line);
            appendSmokeLog(line);
            setDebugHud(String.format(Locale.US, "TOUCH %s %.2f,%.2f result=%d",
                    pressed ? "DOWN" : "UP", x, y, result));
        } catch (Throwable t) {
            Log.i(TAG, "Touchpad input failed: " + t.getMessage());
            setDebugHud("TOUCH failed: " + t.getMessage());
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void openRuntimePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_RUNTIME);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (launcherMode) {
            if (launcherUi != null) {
                launcherUi.handleActivityResult(requestCode, resultCode, data);
            }
            return;
        }
        if (requestCode != PICK_RUNTIME || resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }

        try {
            copyUriToFile(data.getData(), runtimeFile);
            append("Runtime selected: " + describe(data.getData()));
            loadRuntimeFromFile("Selected runtime");
        } catch (Exception e) {
            append("Selected runtime failed: " + e.getMessage());
        }
    }

    private void loadRuntimeFromFile(String label) {
        try {
            append(label + ": " + runtimeFile.getAbsolutePath());
            append(RuntimeBridge.load(runtimeFile.getAbsolutePath()));
            runtimeLoaded = true;
            RuntimeBridge.setAudioEnabled(SettingsActivity.prefs(this).getBoolean(
                    SettingsActivity.K_AUDIO, true));
            replayExternalGamepadState();
            runOnUiThread(this::updateBackendLabel);
            append("ABI: " + RuntimeBridge.abi());
            append("Version: " + RuntimeBridge.version());
            append("System: " + RuntimeBridge.systemInfo());
            attachSurfaceIfReady();
        } catch (Exception e) {
            runtimeLoaded = false;
            append("Runtime load failed: " + e.getMessage());
        }
    }

    private void initializeRuntime() {
        try {
            File root = new File(getFilesDir(), "lsx4-home");
            if (!root.exists() && !root.mkdirs()) {
                throw new IllegalStateException("Cannot create " + root);
            }
            materializeJitPersistentJitCacheSetting(root);
            materializeGraphicsEffectSettings(root);
            materializeRenderResolutionSetting(root);
            writeNativeTranslatorPath(root);
            int result = RuntimeBridge.initialize(root.getAbsolutePath(), "local");
            append("Initialize result: " + result);
            append("Firmware status: " + RuntimeBridge.firmwareStatus());
            append("AArch64 JIT self-test skipped in the interactive app; use the headless native probe.");
            append("Vortek ashmem ring selftest result: " + RuntimeBridge.vortekRingSelfTest());
            append("Audio init result: " + RuntimeBridge.audioInit());
            RuntimeBridge.setAudioEnabled(SettingsActivity.prefs(this).getBoolean(
                    SettingsActivity.K_AUDIO, true));
            setDebugHud("AUDIO init requested");
            if (shouldAutoRunAudioProbe()) {
                runAudioProbeToneAsync();
            } else {
                append("Audio probe tone skipped: live launch owns runtime/audio lifecycle");
            }
            append("Audio status: " + RuntimeBridge.audioStatus());
            append("Runtime status: " + RuntimeBridge.status());
        } catch (Exception e) {
            append("Initialize failed: " + e.getMessage());
        }
    }

    private boolean shouldAutoRunAudioProbe() {
        String launchPath = getIntent().getStringExtra(EXTRA_LAUNCH_GAME_PATH);
        if (launchPath != null && !launchPath.isEmpty()) {
            return false;
        }
        if (intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY) ||
                intentFlag(EXTRA_TRANSLATOR_BRIDGE_LAUNCH) ||
                intentFlag(EXTRA_JAVA_TRANSLATOR_SMOKE)) {
            return false;
        }
        return true;
    }

    private void runAudioProbeToneAsync() {
        Thread thread = new Thread(() -> {
            try {
                int result = RuntimeBridge.audioProbeTone();
                append("Audio probe tone result: " + result);
                append("Audio status: " + RuntimeBridge.audioStatus());
                setDebugHud("AUDIO tone result=" + result);
            } catch (Exception e) {
                append("Audio probe tone failed: " + e.getMessage());
                setDebugHud("AUDIO tone failed: " + e.getMessage());
            }
        }, "ps4run-audio-probe");
        thread.setDaemon(true);
        thread.start();
    }

    private void scanGamePath(String path) {
        try {
            int result = RuntimeBridge.scanGame(path);
            append("Scan game result: " + result);
            append("Runtime status: " + RuntimeBridge.status());
        } catch (Exception e) {
            append("Scan game failed: " + e.getMessage());
        }
    }

    private void launchGamePathJit(String path) {
        try {
            materializeJitPersistentJitCacheSetting(
                    new File(getFilesDir(), "lsx4-home"));
            materializeGraphicsEffectSettings(
                    new File(getFilesDir(), "lsx4-home"));
            materializeRenderResolutionSetting(
                    new File(getFilesDir(), "lsx4-home"));
            append("AArch64 JIT status before launch: " + RuntimeBridge.jitStatus());
            int result = RuntimeBridge.launchGameJit(path);
            append("AArch64 JIT launch result: " + result);
            append("Runtime status: " + RuntimeBridge.status());
            append("AArch64 JIT status after launch: " + RuntimeBridge.jitStatus());
            runOnUiThread(this::updateBackendLabel);
        } catch (Exception e) {
            append("AArch64 JIT launch failed: " + e.getMessage());
        } finally {
            // launchGameJit is synchronous for the guest lifetime. Once it returns (normal
            // exit or -70 frontier fault), the single-task host may accept a deliberate relaunch.
            if (path != null && path.equals(activeGamePath)) {
                activeGamePath = "";
            }
        }
    }

    private void materializeJitPersistentJitCacheSetting(File root) throws Exception {
        if (!root.isDirectory() && !root.mkdirs()) {
            throw new IllegalStateException("Cannot create compiled-block cache root: " + root);
        }
        final boolean requestedByUser = SettingsActivity.prefs(this).getBoolean(
                SettingsActivity.K_PERSISTENT_JIT_CACHE, true);
        final File forceMarker = new File(root, JIT_PERSISTENT_JIT_CACHE_FORCE_MARKER);
        final boolean forcedByAutomation = forceMarker.isFile();
        final boolean enabled = requestedByUser || forcedByAutomation;
        final File marker = new File(root, JIT_PERSISTENT_JIT_CACHE_MARKER);
        try {
            if (enabled) {
                if (marker.exists() && !marker.isFile()) {
                    append("Compiled-block cache unavailable: marker is not a file: " + marker);
                    return;
                }
                if (!marker.isFile()) {
                    writeSmallFile(marker, "1\n");
                }
            } else if (marker.exists() && !marker.delete()) {
                append("Compiled-block cache disabled, but marker could not be removed: " + marker);
                return;
            }
        } catch (Exception e) {
            // The cache is an optional acceleration layer. A storage/marker failure must never
            // prevent the game itself from launching; the native runtime simply sees cache-off.
            append("Compiled-block cache unavailable; continuing without it: " + e.getMessage());
            return;
        }
        append("Persistent compiled block cache: enabled=" + enabled +
                " forced=" + forcedByAutomation +
                " marker=" + marker.getAbsolutePath());
    }

    private void materializeGraphicsEffectSettings(File root) {
        if (!root.isDirectory() && !root.mkdirs()) {
            append("Graphics effect settings unavailable: cannot create " + root);
            return;
        }

        final SharedPreferences preferences = SettingsActivity.prefs(this);
        final String[][] settings = {
                {SettingsActivity.K_DISABLE_DYNAMIC_SHADOWS,
                        DISABLE_DYNAMIC_SHADOWS_MARKER, "dynamic shadows"},
                {SettingsActivity.K_DISABLE_SSAO, DISABLE_SSAO_MARKER, "SSAO"},
                {SettingsActivity.K_DISABLE_MOTION_BLUR,
                        DISABLE_MOTION_BLUR_MARKER, "motion blur"},
                {SettingsActivity.K_DISABLE_DEPTH_OF_FIELD,
                        DISABLE_DEPTH_OF_FIELD_MARKER, "depth of field"},
                {SettingsActivity.K_DISABLE_ANTI_ALIASING,
                        DISABLE_ANTI_ALIASING_MARKER, "anti-aliasing"},
                {SettingsActivity.K_DISABLE_CHROMATIC_ABERRATION,
                        DISABLE_CHROMATIC_ABERRATION_MARKER, "chromatic aberration"}
        };

        for (String[] setting : settings) {
            final boolean disabled = preferences.getBoolean(setting[0], false);
            final File marker = new File(root, setting[1]);
            try {
                if (disabled) {
                    if (marker.exists() && !marker.isFile()) {
                        append("Graphics effect marker is not a file: " + marker);
                        continue;
                    }
                    if (!marker.isFile()) {
                        writeSmallFile(marker, "1\n");
                    }
                } else if (marker.exists() && !marker.delete()) {
                    append("Graphics effect marker could not be removed: " + marker);
                    continue;
                }
                append("Graphics effect setting: " + setting[2] +
                        " disabled=" + disabled +
                        " marker=" + marker.getAbsolutePath());
            } catch (Exception e) {
                // These are optional, title-gated performance patches. A marker/storage problem
                // must never prevent runtime initialization or a guest launch.
                append("Graphics effect setting unavailable for " + setting[2] +
                        "; continuing with the native default: " + e.getMessage());
            }
        }
    }

    /**
     * Publish the user-selected output extent for native/game-specific render-scale consumers.
     * SurfaceHolder.setFixedSize remains the generic presentation path; Java does not pretend that
     * changing a SurfaceView also resizes a title's internal render targets.
     */
    private void materializeRenderResolutionSetting(File root) {
        final int mode = SettingsActivity.prefs(this).getInt(
                SettingsActivity.K_RES_MODE, SettingsActivity.RES_NATIVE);
        final int width = SettingsActivity.resolutionWidth(mode);
        final int height = SettingsActivity.resolutionHeight(mode);
        final File config = new File(root, RENDER_RESOLUTION_CONFIG);
        final File parent = config.getParentFile();
        try {
            if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
                append("Render resolution bridge unavailable: cannot create " + parent);
                return;
            }
            writeSmallFile(config, "version=1\nmode=" + mode + "\nwidth=" + width
                    + "\nheight=" + height + "\n");
            append("Render resolution bridge: mode=" + mode + " extent=" + width + "x"
                    + height + " config=" + config.getAbsolutePath());
        } catch (Exception e) {
            append("Render resolution bridge unavailable; surface extent still applies: "
                    + e.getMessage());
        }
    }

    private void relocateImports() {
        try {
            int result = RuntimeBridge.relocateImports();
            append("Relocate imports result: " + result);
            append("Runtime status: " + RuntimeBridge.status());
            if (result == 0 &&
                    (intentFlag(EXTRA_RUN_EMBEDDED_BOX64_MAPPED_ENTRY) || isLiveGameLaunch())) {
                runEmbeddedBox64MappedEntryOnce("post-relocate");
            }
        } catch (Exception e) {
            append("Relocate imports failed: " + e.getMessage());
        }
    }

    private void prepareBox64EntryTrampoline() {
        try {
            int result = RuntimeBridge.prepareBox64EntryTrampoline();
            append("Prepare Box64 entry trampoline result: " + result);
            append("Box64 entry request: " + RuntimeBridge.box64EntryRequest());
            append("Runtime status: " + RuntimeBridge.status());
        } catch (Exception e) {
            append("Prepare Box64 entry trampoline failed: " + e.getMessage());
        }
    }

    private void runTranslatorGuestElf(String path) {
        try {
            int result = RuntimeBridge.translatorRunGuestElf(path);
            append("Translator guest ELF run result: " + result);
            append("Runtime status: " + RuntimeBridge.status());
            if (intentFlag(EXTRA_PRESENT_FRAME_DUMP_AFTER_GUEST)) {
                presentFrameDumpAfterGuest();
            }
            scheduleSurfacePixelCopy("post_guest_01s", 1000);
            scheduleSurfacePixelCopy("post_guest_03s", 3000);
        } catch (Exception e) {
            append("Translator guest ELF run failed: " + e.getMessage());
        }
    }

    private void probeDlopenPath(String path) {
        try {
            append("External dlopen probe: " + RuntimeBridge.probeDlopen(path));
        } catch (Exception e) {
            append("External dlopen probe failed: " + e.getMessage());
        }
    }

    private void loadEmbeddedBox64Path(String path) {
        try {
            int result = RuntimeBridge.loadEmbeddedBox64(path);
            append("Load embedded Box64 result: " + result);
            append("Embedded Box64 info: " + RuntimeBridge.embeddedBox64Info());
            append("Runtime status: " + RuntimeBridge.status());
        } catch (Exception e) {
            append("Load embedded Box64 failed: " + e.getMessage());
        }
    }

    private void validateEmbeddedBox64MappedEntry() {
        try {
            int result = RuntimeBridge.validateEmbeddedBox64MappedEntry();
            append("Validate embedded Box64 mapped entry result: " + result);
            append("Embedded Box64 mapped entry info: " + RuntimeBridge.embeddedBox64MappedEntryInfo());
            append("Runtime status: " + RuntimeBridge.status());
        } catch (Exception e) {
            append("Validate embedded Box64 mapped entry failed: " + e.getMessage());
        }
    }

    private void runEmbeddedBox64MappedEntry() {
        runEmbeddedBox64MappedEntryOnce("automation");
    }

    private void runEmbeddedBox64MappedEntryOnce(String reason) {
        if (mappedEntryRunRequested) {
            append("Run embedded Box64 mapped entry skipped: already requested reason=" + reason);
            return;
        }
        mappedEntryRunRequested = true;
        resetRuntimeDebugHudForGuestLaunch();
        startRuntimeDebugHud();
        append("Run embedded Box64 mapped entry trigger: " + reason);
        append("Run embedded Box64 mapped entry result: started");
        append("Embedded Box64 mapped entry info: {\"guestEntryAttempted\":true,\"mode\":\"background\",\"next_step\":\"guest is running in-process; capture VideoOut SubmitFlip through Android surface\"}");
        // The LSX4 loading monitor performs a tiny SurfaceView probe and saves one
        // full-size PNG only when the real logo is latched. Avoid the old seven full-resolution
        // PixelCopy+PNG diagnostics during startup; they add avoidable GPU readbacks and I/O exactly
        // while Mono is bootstrapping. Keep them for older non-B diagnostic routes.
        if (!intentFlag(EXTRA_EMBEDDED_AARCH64_JIT_BACKEND)) {
            scheduleSurfacePixelCopy("post_guest_01s", 1000);
            scheduleSurfacePixelCopy("post_guest_03s", 3000);
            scheduleSurfacePixelCopy("post_guest_06s", 6000);
            scheduleSurfacePixelCopy("post_guest_12s", 12000);
            scheduleSurfacePixelCopy("post_guest_20s", 20000);
            scheduleSurfacePixelCopy("post_guest_35s", 35000);
            scheduleSurfacePixelCopy("post_guest_55s", 55000);
        }
        Thread thread = new Thread(() -> {
            try {
                int result = RuntimeBridge.runEmbeddedBox64MappedEntry();
                append("Run embedded Box64 mapped entry completed result: " + result);
                append("Embedded Box64 mapped entry info: " + RuntimeBridge.embeddedBox64MappedEntryInfo());
                append("Runtime status: " + RuntimeBridge.status());
            } catch (Exception e) {
                append("Run embedded Box64 mapped entry failed: " + e.getMessage());
            } finally {
                mappedEntryRunRequested = false;
                activeGamePath = "";
            }
        }, "ps4run-box64-entry");
        thread.setDaemon(true);
        thread.start();
    }

    private void prepareTranslatorLaunchPath(String path) {
        try {
            int result = RuntimeBridge.prepareTranslatorLaunch(path);
            append("Prepare translator launch result: " + result);
            append("Runtime status: " + RuntimeBridge.status());
        } catch (Exception e) {
            append("Prepare translator launch failed: " + e.getMessage());
        }
    }

    private void attachSurfaceIfReady() {
        if (!runtimeLoaded || currentSurface == null || !currentSurface.isValid()) {
            return;
        }

        try {
            int result = RuntimeBridge.attachSurface(currentSurface);
            setSurfaceStatus("Surface attached: " + RuntimeBridge.surfaceInfo());
            append("Attach surface result: " + result);
        } catch (Exception e) {
            setSurfaceStatus("Surface attach failed: " + e.getMessage());
            append("Surface attach failed: " + e.getMessage());
        }
    }

    private void presentTestPattern() {
        if (isLiveGameLaunch()) {
            append("Present test pattern blocked for live game launch.");
            return;
        }
        if (!runtimeLoaded) {
            append("Present test pattern skipped: runtime is not loaded.");
            return;
        }

        try {
            SurfaceView surfaceView = renderSurfaceView;
            int width = surfaceView != null && surfaceView.getWidth() > 0 ? surfaceView.getWidth() : 640;
            int height = surfaceView != null && surfaceView.getHeight() > 0 ? surfaceView.getHeight() : 360;
            int result = RuntimeBridge.presentTestPattern(width, height);
            append("Present test pattern result: " + result);
            append("Surface info: " + RuntimeBridge.surfaceInfo());
            requestSurfacePixelCopy("present_test_pattern");
        } catch (Exception e) {
            append("Present test pattern failed: " + e.getMessage());
        }
    }

    private void presentHomebrewLoaderFrame() {
        if (!runtimeLoaded) {
            append("Present homebrew loader frame skipped: runtime is not loaded.");
            return;
        }

        try {
            SurfaceView surfaceView = renderSurfaceView;
            int width = surfaceView != null && surfaceView.getWidth() > 0 ? surfaceView.getWidth() : 1280;
            int height = surfaceView != null && surfaceView.getHeight() > 0 ? surfaceView.getHeight() : 720;
            int result = RuntimeBridge.presentHomebrewLoaderFrame(width, height);
            append("Present homebrew loader frame result: " + result);
            append("Surface info: " + RuntimeBridge.surfaceInfo());
            append("Runtime status: " + RuntimeBridge.status());
            requestSurfacePixelCopy("homebrew_loader_frame");
        } catch (Exception e) {
            append("Present homebrew loader frame failed: " + e.getMessage());
        }
    }

    private void presentFrameDumpAfterGuest() {
        try {
            File dump = new File(new File(new File(getFilesDir(), "lsx4-home"), "translator"), "executor-videoout-frame.bin");
            int result = RuntimeBridge.presentFrameDumpFile(dump.getAbsolutePath());
            append("Present frame dump result: " + result + " path=" + dump.getAbsolutePath());
            append("Surface info: " + RuntimeBridge.surfaceInfo());
            requestSurfacePixelCopy("present_frame_dump");
        } catch (Exception e) {
            append("Present frame dump failed: " + e.getMessage());
        }
    }

    private void scheduleSurfacePixelCopy(String reason, long delayMs) {
        new Handler(Looper.getMainLooper()).postDelayed(() -> requestSurfacePixelCopy(reason), delayMs);
    }

    /**
     * Samples only the guest SurfaceView, never the Android gamepad/HUD overlay. A 64x36 target is
     * enough to recognize hard logo/text edges and costs 2,304 pixels once per second. A candidate
     * must survive three consecutive probes and a full-size recheck: a one-frame flash must never
     * latch the checkbox or save a later black frame as its evidence.
     */
    private void requestTinyGuestFrameProbe() {
        if (frameCaught || frameDetectionInFlight || !runtimeLoaded) {
            return;
        }
        SurfaceView surfaceView = renderSurfaceView;
        if (surfaceView == null || surfaceView.getWidth() <= 0 || surfaceView.getHeight() <= 0) {
            return;
        }
        Surface surface = surfaceView.getHolder().getSurface();
        if (surface == null || !surface.isValid()) {
            return;
        }

        frameDetectionInFlight = true;
        Bitmap probe = Bitmap.createBitmap(64, 36, Bitmap.Config.ARGB_8888);
        try {
            PixelCopy.request(surfaceView, probe, result -> {
                frameDetectionInFlight = false;
                if (result == PixelCopy.SUCCESS && !frameCaught) {
                    int[] pixels = new int[64 * 36];
                    probe.getPixels(pixels, 0, 64, 0, 0, 64, 36);
                    int[] score = scoreDetailedGuestFrame(pixels, 64, 36);
                    frameProbeNonblack = score[0];
                    frameProbeEdges = score[1];
                    if (score[2] != 0) {
                        frameProbeAcceptedStreak = Math.min(FRAME_PROBE_REQUIRED_STREAK,
                                frameProbeAcceptedStreak + 1);
                    } else {
                        frameProbeAcceptedStreak = 0;
                    }
                    if (frameProbeAcceptedStreak >= FRAME_PROBE_REQUIRED_STREAK) {
                        requestVerifiedCaughtGuestFrameEvidence();
                    }
                }
                probe.recycle();
            }, runtimeHudHandler);
        } catch (Throwable t) {
            frameDetectionInFlight = false;
            probe.recycle();
        }
    }

    // Returns {meaningful non-black pixels, strong local edges, accepted}.  At the downscaled probe
    // resolution a smooth flash changes only a few RGB levels between neighbours, while logo glyphs
    // retain many high-frequency, high-contrast edges.
    private int[] scoreDetailedGuestFrame(int[] pixels, int width, int height) {
        int meaningful = 0;
        int edges = 0;
        int veryStrongEdges = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int pixel = pixels[index];
                int r = (pixel >>> 16) & 0xff;
                int g = (pixel >>> 8) & 0xff;
                int b = pixel & 0xff;
                int luma = (r * 54 + g * 183 + b * 19) >>> 8;
                if (luma > 12 || Math.max(r, Math.max(g, b)) > 20) {
                    meaningful++;
                }
                if (x > 0) {
                    int edge = pixelEdgeStrength(pixel, pixels[index - 1]);
                    if (edge >= 48) edges++;
                    if (edge >= 86) veryStrongEdges++;
                }
                if (y > 0) {
                    int edge = pixelEdgeStrength(pixel, pixels[index - width]);
                    if (edge >= 48) edges++;
                    if (edge >= 86) veryStrongEdges++;
                }
            }
        }
        // The thresholds are intentionally absolute at the fixed 64x36 probe size.  They require
        // visible content plus spatial detail; neither a black frame nor the observed smooth bloom
        // can latch the checkbox.
        boolean accepted = meaningful >= 28 && edges >= 18 && veryStrongEdges >= 6;
        return new int[]{meaningful, edges, accepted ? 1 : 0};
    }

    private int pixelEdgeStrength(int a, int b) {
        int dr = Math.abs(((a >>> 16) & 0xff) - ((b >>> 16) & 0xff));
        int dg = Math.abs(((a >>> 8) & 0xff) - ((b >>> 8) & 0xff));
        int db = Math.abs((a & 0xff) - (b & 0xff));
        return Math.max(dr, Math.max(dg, db));
    }

    private void latchCaughtGuestFrame(Bitmap verifiedFrame) {
        if (frameCaught) {
            verifiedFrame.recycle();
            return;
        }
        frameCaught = true;
        if (loadingProgressPanel != null) {
            loadingProgressPanel.setVisibility(View.GONE);
        }
        frameCaughtAtMs = Math.max(0, SystemClock.elapsedRealtime() - runtimeHudStartedAtMs);
        long caughtSeconds = frameCaughtAtMs / 1000;
        if (frameCaughtCheck != null) {
            frameCaughtCheck.setChecked(true);
            frameCaughtCheck.setText(getString(R.string.hud_frame_verified,
                    caughtSeconds / 60, caughtSeconds % 60));
        }
        runtimeHudEvent = "stable guest frame verified: 3 probes + full-size recheck";
        renderRuntimeDebugHud();
        Thread saver = new Thread(() -> {
            String path = savePixelCopyFrame(verifiedFrame, "auto-verified-guest-frame");
            verifiedFrame.recycle();
            runOnUiThread(() -> {
                frameCaughtPath = path;
                // Keep the automatic verified-frame path in the same machine-readable stream as
                // explicit PixelCopy requests.  The smoke harness intentionally never falls back
                // to Android screencap for a live Vulkan guest, so without this record it cannot
                // pull the valid app-owned PNG that the detector just saved.
                append("PixelCopy result: {\"attempted\":true,\"success\":" +
                        (!path.isEmpty()) +
                        ",\"trigger\":\"auto_verified_guest_frame\",\"framePath\":\"" +
                        sanitizeJson(path) + "\"}");
                runtimeHudEvent = path.isEmpty()
                        ? "verified frame latched; PNG save failed"
                        : "verified frame PNG saved";
                renderRuntimeDebugHud();
            });
        }, "ps4run-verified-frame-save");
        saver.setDaemon(true);
        saver.start();
    }

    /** One full-resolution copy verifies the stable tiny-probe candidate and becomes its evidence. */
    private void requestVerifiedCaughtGuestFrameEvidence() {
        if (frameCaught || frameDetectionInFlight) {
            return;
        }
        SurfaceView surfaceView = renderSurfaceView;
        if (surfaceView == null || surfaceView.getWidth() <= 0 || surfaceView.getHeight() <= 0) {
            return;
        }
        Surface surface = surfaceView.getHolder().getSurface();
        if (surface == null || !surface.isValid()) {
            return;
        }
        frameDetectionInFlight = true;
        try {
            Bitmap bitmap = Bitmap.createBitmap(surfaceView.getWidth(), surfaceView.getHeight(),
                    Bitmap.Config.ARGB_8888);
            PixelCopy.request(surfaceView, bitmap, result -> {
                frameDetectionInFlight = false;
                if (result != PixelCopy.SUCCESS) {
                    bitmap.recycle();
                    frameProbeAcceptedStreak = 0;
                    runtimeHudEvent = "stable frame full-size verification failed=" + result;
                    renderRuntimeDebugHud();
                    return;
                }
                Bitmap verification = Bitmap.createScaledBitmap(bitmap, 64, 36, true);
                int[] pixels = new int[64 * 36];
                verification.getPixels(pixels, 0, 64, 0, 0, 64, 36);
                verification.recycle();
                int[] score = scoreDetailedGuestFrame(pixels, 64, 36);
                frameProbeNonblack = score[0];
                frameProbeEdges = score[1];
                if (score[2] == 0) {
                    bitmap.recycle();
                    frameProbeAcceptedStreak = 0;
                    runtimeHudEvent = "candidate rejected by full-size recheck";
                    renderRuntimeDebugHud();
                    return;
                }
                latchCaughtGuestFrame(bitmap);
            }, runtimeHudHandler);
        } catch (Throwable t) {
            frameDetectionInFlight = false;
            frameProbeAcceptedStreak = 0;
            runtimeHudEvent = "stable frame verification error=" + t.getClass().getSimpleName();
        }
    }

    private void requestSurfacePixelCopy(String reason) {
        SurfaceView surfaceView = renderSurfaceView;
        if (surfaceView == null || surfaceView.getWidth() <= 0 || surfaceView.getHeight() <= 0) {
            append("PixelCopy result: {\"attempted\":false,\"reason\":\"surface_unavailable\",\"trigger\":\"" + sanitizeJson(reason) + "\"}");
            return;
        }

        runOnUiThread(() -> {
            try {
                Surface surface = surfaceView.getHolder().getSurface();
                if (surface == null || !surface.isValid()) {
                    append("PixelCopy result: {\"attempted\":false,\"reason\":\"surface_unavailable\",\"trigger\":\"" + sanitizeJson(reason) + "\"}");
                    return;
                }

                Bitmap bitmap = Bitmap.createBitmap(surfaceView.getWidth(), surfaceView.getHeight(), Bitmap.Config.ARGB_8888);
                PixelCopy.request(surfaceView, bitmap, copyResult -> {
                    long sampled = 0;
                    long nonzero = 0;
                    int firstNonzero = 0;
                    if (copyResult == PixelCopy.SUCCESS) {
                        int width = bitmap.getWidth();
                        int height = bitmap.getHeight();
                        int[] row = new int[width];
                        for (int y = 0; y < height; y++) {
                            bitmap.getPixels(row, 0, width, 0, y, width, 1);
                            for (int x = 0; x < width; x++) {
                                sampled++;
                                int pixel = row[x];
                                if ((pixel & 0x00ffffff) == 0) {
                                    continue;
                                }
                                if (nonzero == 0) {
                                    firstNonzero = pixel;
                                }
                                nonzero++;
                            }
                        }
                    }

                    String framePath = "";
                    if (copyResult == PixelCopy.SUCCESS) {
                        framePath = savePixelCopyFrame(bitmap, reason);
                    }

                    String json = "{\"attempted\":true" +
                            ",\"trigger\":\"" + sanitizeJson(reason) + "\"" +
                            ",\"result\":" + copyResult +
                            ",\"success\":" + (copyResult == PixelCopy.SUCCESS) +
                            ",\"width\":" + bitmap.getWidth() +
                            ",\"height\":" + bitmap.getHeight() +
                            ",\"sampledPixels\":" + sampled +
                            ",\"nonzeroPixels\":" + nonzero +
                            ",\"firstNonzeroPixel\":\"" + String.format(Locale.US, "%08X", firstNonzero) + "\"" +
                            ",\"framePath\":\"" + sanitizeJson(framePath) + "\"}";
                    append("PixelCopy result: " + json);
                    bitmap.recycle();
                }, new Handler(Looper.getMainLooper()));
            } catch (Exception e) {
                append("PixelCopy result: {\"attempted\":true,\"success\":false,\"trigger\":\"" + sanitizeJson(reason) + "\",\"error\":\"" + sanitizeJson(e.getMessage()) + "\"}");
            }
        });
    }

    private String savePixelCopyFrame(Bitmap bitmap, String reason) {
        File directory = new File(getFilesDir(), "pixelcopy");
        if (!directory.exists() && !directory.mkdirs()) {
            return "";
        }

        String safeReason = reason == null ? "frame" : reason.replaceAll("[^A-Za-z0-9_.-]", "_");
        File file = new File(directory, String.format(Locale.US, "ps4run-pixelcopy-frame-%02d-%s.png", pixelCopyFrameIndex++, safeReason));
        try (FileOutputStream out = new FileOutputStream(file)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                return "";
            }
            return file.getAbsolutePath();
        } catch (Exception e) {
            append("PixelCopy frame save failed: " + sanitizeJson(e.getMessage()));
            return "";
        }
    }

    private void detachSurface() {
        if (!runtimeLoaded) {
            setSurfaceStatus(getString(R.string.surface_not_attached));
            return;
        }

        try {
            RuntimeBridge.detachSurface();
            setSurfaceStatus(getString(R.string.surface_detached));
            append("Surface detached.");
        } catch (Exception e) {
            setSurfaceStatus(getString(R.string.surface_detach_failed, e.getMessage()));
        }
    }

    private void setSurfaceStatus(String text) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            runOnUiThread(() -> setSurfaceStatus(text));
            return;
        }
        if (surfaceStatus != null) {
            surfaceStatus.setText(text);
        }
    }

    private void writeNativeTranslatorPath(File root) throws Exception {
        File translatorDir = new File(root, "translator");
        if (!translatorDir.exists() && !translatorDir.mkdirs()) {
            throw new IllegalStateException("Cannot create " + translatorDir);
        }

        File nativeBox64 = new File(getApplicationInfo().nativeLibraryDir, "libbox64_executor.so");
        if (!nativeBox64.isFile()) {
            return;
        }

        File pathFile = new File(translatorDir, "box64.path");
        try (FileOutputStream out = new FileOutputStream(pathFile)) {
            out.write(nativeBox64.getAbsolutePath().getBytes(StandardCharsets.UTF_8));
        }
    }

    private void installWinlatorBox64(File root) {
        try {
            File staged = new File(new File(root, "translator-assets"), "box64");
            for (String asset : WINLATOR_BOX64_ASSETS) {
                copyAssetToFile("box64/" + asset, new File(staged, asset));
            }
            writeSmallFile(new File(staged, "native-library-dir.txt"), getApplicationInfo().nativeLibraryDir);

            int installResult = RuntimeBridge.installTranslatorHelper(staged.getAbsolutePath());
            append("Translator helper install result: " + installResult + " source=" + staged.getAbsolutePath());

            File libBox64 = new File(new File(getFilesDir(), "box64/lib"), "libbox64.so");
            if (libBox64.isFile()) {
                append("Box64 dlopen probe: " + RuntimeBridge.probeDlopen(libBox64.getAbsolutePath()));
            } else {
                append("Box64 dlopen probe: missing " + libBox64.getAbsolutePath());
            }
        } catch (Exception e) {
            append("Translator helper install failed: " + e.getMessage());
        }
    }

    private void copyAssetToFile(String assetPath, File target) throws Exception {
        File dir = target.getParentFile();
        if (dir != null && !dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Cannot create " + dir);
        }
        try (InputStream in = getAssets().open(assetPath);
             FileOutputStream out = new FileOutputStream(target)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
        }
    }

    private void writeSmallFile(File target, String value) throws Exception {
        File dir = target.getParentFile();
        if (dir != null && !dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Cannot create " + dir);
        }
        try (FileOutputStream out = new FileOutputStream(target)) {
            out.write(value.getBytes(StandardCharsets.UTF_8));
        }
    }

    private void runTranslatorBridgeHelper() {
        try {
            File root = new File(getFilesDir(), "lsx4-home");
            File nativeLibraryDir = new File(getApplicationInfo().nativeLibraryDir);
            TranslatorBridgeLauncher.Result result = TranslatorBridgeLauncher.launch(root, nativeLibraryDir);
            append(result.requestLine);
            append(result.helperLine);
            if (result.reported) {
                append("Report translator result: " + result.reportResult);
                append("Runtime status: " + RuntimeBridge.status());
            }
        } catch (Exception e) {
            append("Translator bridge helper failed: " + e.getMessage());
        }
    }

    private void copyUriToFile(Uri uri, File target) throws Exception {
        File dir = target.getParentFile();
        if (dir != null && !dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Cannot create " + dir);
        }

        try (InputStream in = getContentResolver().openInputStream(uri);
             FileOutputStream out = new FileOutputStream(target)) {
            if (in == null) {
                throw new IllegalStateException("Cannot open " + uri);
            }
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
        }
    }

    private String describe(Uri uri) {
        try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        } catch (Exception ignored) {
        }
        return uri.toString();
    }

    private String sanitizeJson(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }

    private void append(String line) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            runOnUiThread(() -> append(line));
            return;
        }
        if (output != null) {
            output.append("\n\n" + line);
        }
        Log.i(TAG, line);
        appendSmokeLog(line);
    }

    private void loadStoreCatalogPreview() {
        StoreCatalogView preview = storeCatalogPreview;
        if (preview == null) {
            return;
        }

        File db = new File(new File(new File(new File(new File(getFilesDir(), "lsx4-home"),
                "runtime-fs"), "user/app"), "NPXS39041"), "store.db");
        preview.setThemeAssets(
                decodeIfExists(new File(new File(new File(getFilesDir(), "lsx4-home"),
                        "runtime-fs/app0/assets"), "backgroundUnitOffline.png")),
                decodeIfExists(new File(new File(new File(getFilesDir(), "lsx4-home"),
                        "runtime-fs/app0/sce_sys"), "icon0.png")),
                decodeIfExists(new File(new File(new File(getFilesDir(), "lsx4-home"),
                        "runtime-fs/app0/assets"), "aaa.png")));

        ArrayList<StoreItem> items = new ArrayList<>();
        if (db.isFile()) {
            try (SQLiteDatabase database = SQLiteDatabase.openDatabase(db.getAbsolutePath(), null,
                    SQLiteDatabase.OPEN_READONLY | SQLiteDatabase.NO_LOCALIZED_COLLATORS);
                 Cursor cursor = database.rawQuery(
                         "SELECT name, Author, version, Size, apptype, releaseddate, " +
                                 "main_menu_pic, picpath, main_icon_path " +
                                 "FROM homebrews ORDER BY CAST(pid AS INTEGER) LIMIT 300",
                         null)) {
                while (cursor.moveToNext()) {
                    Bitmap cover = decodeGuestPath(safeColumn(cursor, 6));
                    if (cover == null) {
                        cover = decodeGuestPath(safeColumn(cursor, 7));
                    }
                    if (cover == null) {
                        cover = decodeGuestPath(safeColumn(cursor, 8));
                    }
                    items.add(new StoreItem(
                            safeColumn(cursor, 0),
                            safeColumn(cursor, 1),
                            safeColumn(cursor, 2),
                            safeColumn(cursor, 3),
                            safeColumn(cursor, 4),
                            safeColumn(cursor, 5),
                            cover));
                }
            } catch (Exception e) {
                append("Store catalog preview DB load failed: " + e.getMessage());
            }
        }
        if (items.isEmpty()) {
            items.add(new StoreItem("Itemzflow Game Manager", "LM", "1.07", "25.50 MB", "Utility", "2025-06-22", null));
            items.add(new StoreItem("PS4-Xplorer 2.0", "Lapy", "2.07", "60.81 MB", "Utility", "2024-05-10", null));
            items.add(new StoreItem("Apollo Save Tool", "bucanero", "2.3.2", "21.63 MB", "Utility", "2024-05-26", null));
            items.add(new StoreItem("PS4-Xplorer", "Lapy", "1.34", "41.10 MB", "Utility", "2023-11-09", null));
        }
        preview.setItems(items);
        append("Store catalog preview loaded: items=" + items.size() + " db=" + db.getAbsolutePath());
        preview.post(() -> saveStoreCatalogPreviewFrame("store_catalog_preview_ready"));
    }

    private Bitmap decodeGuestPath(String guestPath) {
        if (guestPath == null || guestPath.trim().isEmpty()) {
            return null;
        }
        String normalized = guestPath.trim();
        if (!normalized.startsWith("/")) {
            return decodeIfExists(new File(normalized));
        }
        File root = new File(new File(getFilesDir(), "lsx4-home"), "runtime-fs");
        return decodeIfExists(new File(root, normalized.substring(1)));
    }

    private Bitmap decodeIfExists(File file) {
        if (file == null || !file.isFile()) {
            return null;
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeFile(file.getAbsolutePath(), options);
    }

    private String safeColumn(Cursor cursor, int index) {
        if (cursor.isNull(index)) {
            return "";
        }
        return cursor.getString(index);
    }

    private void saveStoreCatalogPreviewFrame(String reason) {
        StoreCatalogView preview = storeCatalogPreview;
        SurfaceView surfaceView = renderSurfaceView;
        if (preview == null || surfaceView == null ||
                surfaceView.getWidth() <= 0 || surfaceView.getHeight() <= 0) {
            return;
        }
        try {
            Bitmap bitmap = Bitmap.createBitmap(surfaceView.getWidth(), surfaceView.getHeight(),
                    Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bitmap);
            preview.layout(0, 0, bitmap.getWidth(), bitmap.getHeight());
            preview.draw(canvas);
            String framePath = savePixelCopyFrame(bitmap, reason);
            append("Store catalog preview frame: {\"framePath\":\"" + sanitizeJson(framePath) +
                    "\",\"width\":" + bitmap.getWidth() +
                    ",\"height\":" + bitmap.getHeight() +
                    ",\"items\":" + preview.getItemCount() + "}");
            bitmap.recycle();
        } catch (Exception e) {
            append("Store catalog preview frame failed: " + e.getMessage());
        }
    }

    private void appendSmokeLog(String line) {
        File home = new File(getFilesDir(), "lsx4-home");
        if (!home.exists() && !home.mkdirs()) {
            return;
        }

        File log = new File(home, "ui-smoke.log");
        try (FileOutputStream out = new FileOutputStream(log, true)) {
            out.write((line + "\n").getBytes(StandardCharsets.UTF_8));
        } catch (Exception ignored) {
        }
    }

    private static final class StoreItem {
        final String name;
        final String author;
        final String version;
        final String size;
        final String type;
        final String released;
        final Bitmap cover;

        StoreItem(String name, String author, String version, String size, String type,
                  String released, Bitmap cover) {
            this.name = name == null || name.isEmpty() ? "Homebrew" : name;
            this.author = author == null ? "" : author;
            this.version = version == null ? "" : version;
            this.size = size == null ? "" : size;
            this.type = type == null ? "" : type;
            this.released = released == null ? "" : released;
            this.cover = cover;
        }
    }

    private interface StoreCatalogActionListener {
        void onExecute(StoreItem item);
    }

    private static final class StoreCatalogView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final RectF rect = new RectF();
        private final RectF touchRect = new RectF();
        private final Rect srcRect = new Rect();
        private final ArrayList<StoreItem> items = new ArrayList<>();
        private final ArrayList<StoreItem> allItems = new ArrayList<>();
        private static final String[] TABS = {"All", "Games", "Apps", "Emulators", "Media"};
        private final RectF[] tabRects = new RectF[TABS.length];
        private int selectedTab;
        private Bitmap background;
        private Bitmap storeIcon;
        private Bitmap coverFrame;
        private int selectedIndex;
        private int lastColumns = 4;
        private int touchDownIndex = -1;
        private int touchDownTab = -1;
        private StoreItem detailItem;
        private StoreCatalogActionListener actionListener;
        private String statusLine = "Select a tab • tap a tile to open";

        StoreCatalogView(Context context) {
            super(context);
            setWillNotDraw(false);
            setClickable(true);
            // The PS4 Store's nav SFX are PS4 system (firmware) sounds, not app assets; the homebrew
            // ships no audio. Silence Android's default click sound rather than play an inauthentic one.
            setSoundEffectsEnabled(false);
            for (int i = 0; i < tabRects.length; i++) {
                tabRects[i] = new RectF();
            }
        }

        void setActionListener(StoreCatalogActionListener listener) {
            actionListener = listener;
        }

        void setThemeAssets(Bitmap background, Bitmap storeIcon, Bitmap coverFrame) {
            this.background = background;
            this.storeIcon = storeIcon;
            this.coverFrame = coverFrame;
            invalidate();
        }

        void setItems(List<StoreItem> value) {
            allItems.clear();
            allItems.addAll(value);
            applyFilter();
        }

        private void applyFilter() {
            items.clear();
            for (StoreItem item : allItems) {
                if (categoryMatches(selectedTab, item.type)) {
                    items.add(item);
                }
            }
            selectedIndex = Math.max(0, Math.min(selectedIndex, Math.max(0, items.size() - 1)));
            invalidate();
        }

        private boolean categoryMatches(int tab, String apptype) {
            String t = apptype == null ? "" : apptype.trim().toLowerCase();
            switch (tab) {
                case 1: // Games
                    return t.contains("game");
                case 2: // Apps (utilities / dev menu / themes / mods / generic homebrew)
                    return t.contains("util") || t.contains("dev") || t.contains("theme")
                            || t.contains("mod") || t.equals("homebrew");
                case 3: // Emulators
                    return t.contains("emulator") || t.contains("emu");
                case 4: // Media
                    return t.contains("media");
                default: // All
                    return true;
            }
        }

        int getItemCount() {
            return items.size();
        }

        private void selectTab(int tab) {
            if (tab < 0 || tab >= TABS.length || tab == selectedTab) {
                return;
            }
            selectedTab = tab;
            selectedIndex = 0;
            applyFilter();
            statusLine = TABS[tab] + " • " + items.size() + " item" + (items.size() == 1 ? "" : "s");
        }

        String handlePad(int mask) {
            // Detail overlay is modal: X = open/install action, O = back to the grid.
            if (detailItem != null) {
                if (mask == PAD_CROSS) {
                    StoreItem item = detailItem;
                    if (actionListener != null) {
                        actionListener.onExecute(item);
                    }
                    statusLine = "Opening: " + item.name;
                    invalidate();
                    return statusLine;
                } else if (mask == PAD_CIRCLE) {
                    detailItem = null;
                    statusLine = "Back to " + TABS[selectedTab];
                    invalidate();
                    return statusLine;
                }
                return null;
            }

            if (mask == PAD_L1) {
                selectTab(selectedTab - 1);
                invalidate();
                return statusLine;
            } else if (mask == PAD_R1) {
                selectTab(selectedTab + 1);
                invalidate();
                return statusLine;
            }

            if (items.isEmpty()) {
                statusLine = TABS[selectedTab] + " • no items (L1/R1 to switch tab)";
                invalidate();
                return statusLine;
            }

            if (mask == PAD_RIGHT) {
                moveSelection(1);
            } else if (mask == PAD_LEFT) {
                moveSelection(-1);
            } else if (mask == PAD_DOWN) {
                moveSelection(Math.max(1, lastColumns));
            } else if (mask == PAD_UP) {
                moveSelection(-Math.max(1, lastColumns));
            } else if (mask == PAD_CROSS) {
                detailItem = items.get(selectedIndex);
                statusLine = "Open: " + detailItem.name + "  (X install • O back)";
                invalidate();
                return statusLine;
            } else if (mask == PAD_OPTIONS) {
                statusLine = "Options";
                invalidate();
                return statusLine;
            } else {
                return null;
            }

            StoreItem item = items.get(selectedIndex);
            statusLine = "Selected: " + item.name;
            invalidate();
            return statusLine;
        }

        private void moveSelection(int delta) {
            selectedIndex = Math.max(0, Math.min(items.size() - 1, selectedIndex + delta));
        }

        @Override
        public boolean onTouchEvent(MotionEvent event) {
            if (items.isEmpty()) {
                return false;
            }
            int action = event.getActionMasked();
            if (action == MotionEvent.ACTION_DOWN) {
                touchDownIndex = hitCardIndex(event.getX(), event.getY());
                if (touchDownIndex < 0) {
                    return false;
                }
                selectedIndex = touchDownIndex;
                statusLine = "Selected: " + items.get(selectedIndex).name;
                invalidate();
                return true;
            }
            if (action == MotionEvent.ACTION_UP) {
                int upIndex = hitCardIndex(event.getX(), event.getY());
                if (touchDownIndex >= 0 && upIndex == touchDownIndex) {
                    selectedIndex = upIndex;
                    StoreItem item = items.get(selectedIndex);
                    statusLine = "X sent: " + item.name;
                    if (actionListener != null) {
                        actionListener.onExecute(item);
                    }
                    performClick();
                }
                touchDownIndex = -1;
                invalidate();
                return true;
            }
            if (action == MotionEvent.ACTION_CANCEL) {
                touchDownIndex = -1;
                invalidate();
                return true;
            }
            return true;
        }

        @Override
        public boolean performClick() {
            super.performClick();
            return true;
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            int width = getWidth();
            int height = getHeight();
            if (width <= 0 || height <= 0) {
                return;
            }

            drawBackground(canvas, width, height);
            drawHeader(canvas, width);
            drawCatalog(canvas, width, height);
            drawStatus(canvas, width, height);
        }

        private void drawBackground(Canvas canvas, int width, int height) {
            if (background != null) {
                drawBitmapCover(canvas, background, 0, 0, width, height, 255);
            } else {
                canvas.drawColor(0xff0750b8);
            }
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(0xaa00102c);
            canvas.drawRect(0, 0, width, height, paint);
            paint.setColor(0x66000000);
            rect.set(0, 0, width, scale(154));
            canvas.drawRect(rect, paint);
        }

        private void drawHeader(Canvas canvas, int width) {
            if (storeIcon != null) {
                drawBitmapCover(canvas, storeIcon, scale(50), scale(33), scale(92), scale(92), 255);
            }
            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
            paint.setColor(0xfff3f7fb);
            paint.setTextSize(scale(42));
            canvas.drawText("Homebrew Store", storeIcon != null ? scale(160) : scale(58), scale(78), paint);

            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.NORMAL));
            paint.setColor(0xffd8e9ff);
            paint.setTextSize(scale(18));
            canvas.drawText("Utilities", scale(58), scale(128), paint);
            paint.setColor(0x80d8e9ff);
            canvas.drawText("Media", scale(178), scale(128), paint);
            canvas.drawText("Emulators", scale(282), scale(128), paint);

            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
            paint.setColor(0xeeffffff);
            paint.setTextSize(scale(21));
            canvas.drawText("PS4", width - scale(120), scale(78), paint);
        }

        private void drawCatalog(Canvas canvas, int width, int height) {
            int columns = width >= 2500 ? 6 : width >= 1700 ? 5 : 4;
            lastColumns = columns;
            float gap = scale(24);
            float left = scale(56);
            float top = scale(178);
            float cardWidth = (width - left * 2f - gap * (columns - 1)) / columns;
            float cardHeight = Math.max(scale(236), (height - top - scale(52) - gap * 2f) / 3f);

            int count = Math.min(items.size(), columns * 3);
            for (int i = 0; i < count; i++) {
                int column = i % columns;
                int row = i / columns;
                float x = left + column * (cardWidth + gap);
                float y = top + row * (cardHeight + gap);
                drawCard(canvas, items.get(i), i, x, y, cardWidth, cardHeight);
            }
        }

        private int hitCardIndex(float px, float py) {
            int width = getWidth();
            int height = getHeight();
            if (width <= 0 || height <= 0) {
                return -1;
            }
            int columns = width >= 2500 ? 6 : width >= 1700 ? 5 : 4;
            float gap = scale(24);
            float left = scale(56);
            float top = scale(178);
            float cardWidth = (width - left * 2f - gap * (columns - 1)) / columns;
            float cardHeight = Math.max(scale(236), (height - top - scale(52) - gap * 2f) / 3f);
            int count = Math.min(items.size(), columns * 3);
            for (int i = 0; i < count; i++) {
                int column = i % columns;
                int row = i / columns;
                float x = left + column * (cardWidth + gap);
                float y = top + row * (cardHeight + gap);
                touchRect.set(x, y, x + cardWidth, y + cardHeight);
                if (touchRect.contains(px, py)) {
                    return i;
                }
            }
            return -1;
        }

        private void drawCard(Canvas canvas, StoreItem item, int index, float x, float y,
                              float width, float height) {
            int accent = palette(index);
            boolean selected = index == selectedIndex;
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(selected ? 0xee1e77db : 0xdd10285f);
            rect.set(x, y, x + width, y + height);
            canvas.drawRoundRect(rect, scale(6), scale(6), paint);
            if (selected) {
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(scale(4));
                paint.setColor(0xfffff2a8);
                canvas.drawRoundRect(rect, scale(6), scale(6), paint);
                paint.setStyle(Paint.Style.FILL);
            }

            paint.setColor(0x55000000);
            rect.set(x + scale(4), y + height - scale(74), x + width - scale(4), y + height - scale(4));
            canvas.drawRoundRect(rect, scale(4), scale(4), paint);

            float coverX = x + scale(13);
            float coverY = y + scale(12);
            float coverW = width - scale(26);
            float coverH = height * 0.56f;
            drawCover(canvas, item, index, coverX, coverY, coverW, coverH, accent);

            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
            paint.setTextSize(scale(index == 0 ? 23 : 20));
            paint.setColor(0xfff2f5f7);
            drawClippedText(canvas, item.name, x + scale(16), y + height * 0.67f,
                    width - scale(32), paint);

            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.NORMAL));
            paint.setTextSize(scale(16));
            paint.setColor(0xffd4e7ff);
            String meta = compactMeta(item);
            drawClippedText(canvas, meta, x + scale(16), y + height * 0.79f,
                    width - scale(32), paint);

            paint.setTextSize(scale(15));
            paint.setColor(0xffa9c6e8);
            drawClippedText(canvas, item.type, x + scale(16), y + height - scale(28),
                    width - scale(32), paint);
        }

        private void drawCover(Canvas canvas, StoreItem item, int index, float x, float y,
                               float width, float height, int accent) {
            if (coverFrame != null) {
                drawBitmapCover(canvas, coverFrame, x, y, width, height, 230);
            } else {
                paint.setColor(0xdd061736);
                rect.set(x, y, x + width, y + height);
                canvas.drawRoundRect(rect, scale(4), scale(4), paint);
            }

            float inset = scale(12);
            if (item.cover != null) {
                drawBitmapCover(canvas, item.cover, x + inset, y + inset,
                        width - inset * 2f, height - inset * 2f, 255);
                return;
            }

            paint.setColor(accent);
            rect.set(x + inset, y + inset, x + width - inset, y + height - inset);
            canvas.drawRoundRect(rect, scale(3), scale(3), paint);
            paint.setColor(0x44ffffff);
            canvas.drawRect(x + inset, y + inset, x + width - inset, y + height * 0.40f, paint);
            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
            paint.setColor(0xffffffff);
            paint.setTextSize(scale(36));
            canvas.drawText(initials(item.name), x + inset + scale(18), y + inset + scale(54), paint);
        }

        private void drawBitmapCover(Canvas canvas, Bitmap bitmap, float x, float y, float width,
                                     float height, int alpha) {
            if (bitmap == null || bitmap.isRecycled()) {
                return;
            }
            float sourceRatio = bitmap.getWidth() / (float) bitmap.getHeight();
            float targetRatio = width / height;
            if (sourceRatio > targetRatio) {
                int sourceWidth = Math.max(1, Math.round(bitmap.getHeight() * targetRatio));
                int left = Math.max(0, (bitmap.getWidth() - sourceWidth) / 2);
                srcRect.set(left, 0, Math.min(bitmap.getWidth(), left + sourceWidth), bitmap.getHeight());
            } else {
                int sourceHeight = Math.max(1, Math.round(bitmap.getWidth() / targetRatio));
                int top = Math.max(0, (bitmap.getHeight() - sourceHeight) / 2);
                srcRect.set(0, top, bitmap.getWidth(), Math.min(bitmap.getHeight(), top + sourceHeight));
            }
            rect.set(x, y, x + width, y + height);
            int oldAlpha = paint.getAlpha();
            paint.setAlpha(alpha);
            canvas.drawBitmap(bitmap, srcRect, rect, paint);
            paint.setAlpha(oldAlpha);
        }

        private void drawClippedText(Canvas canvas, String text, float x, float y, float maxWidth,
                                     Paint textPaint) {
            String value = text == null ? "" : text.trim();
            if (value.isEmpty()) {
                return;
            }
            if (textPaint.measureText(value) > maxWidth) {
                final String ellipsis = "...";
                int end = value.length();
                while (end > 0 &&
                        textPaint.measureText(value.substring(0, end).trim() + ellipsis) > maxWidth) {
                    end--;
                }
                if (end <= 0) {
                    return;
                }
                value = value.substring(0, end).trim() + ellipsis;
            }
            canvas.drawText(value, x, y, textPaint);
        }

        private void drawStatus(Canvas canvas, int width, int height) {
            if (statusLine == null || statusLine.isEmpty()) {
                return;
            }
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(0xaa00102c);
            rect.set(scale(52), height - scale(54), width - scale(52), height - scale(18));
            canvas.drawRoundRect(rect, scale(6), scale(6), paint);
            paint.setTypeface(Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD));
            paint.setTextSize(scale(18));
            paint.setColor(0xfff5f8ff);
            drawClippedText(canvas, statusLine, scale(70), height - scale(30),
                    width - scale(140), paint);
        }

        private String compactMeta(StoreItem item) {
            StringBuilder out = new StringBuilder();
            if (!item.version.isEmpty()) {
                out.append('v').append(item.version);
            }
            if (!item.size.isEmpty()) {
                if (out.length() > 0) {
                    out.append("  ");
                }
                out.append(item.size);
            }
            if (!item.author.isEmpty()) {
                if (out.length() > 0) {
                    out.append("  ");
                }
                out.append(item.author);
            }
            return out.toString();
        }

        private String initials(String name) {
            String[] parts = name.trim().split("\\s+");
            StringBuilder out = new StringBuilder();
            for (String part : parts) {
                if (part.isEmpty()) {
                    continue;
                }
                out.append(Character.toUpperCase(part.charAt(0)));
                if (out.length() >= 3) {
                    break;
                }
            }
            return out.length() == 0 ? "HB" : out.toString();
        }

        private int palette(int index) {
            final int[] colors = {
                    0xff3273dc, 0xff1f9d7a, 0xffd14f4f, 0xff8a6bd8,
                    0xffd39a28, 0xff2584a6, 0xff6a8d3a, 0xffb14d8b
            };
            return colors[index % colors.length];
        }

        private float scale(float value) {
            return value * Math.max(0.72f, Math.min(1.6f, getHeight() / 1080f));
        }
    }
}
