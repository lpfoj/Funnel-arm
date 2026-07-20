package app.lsx4.android;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.StateListDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.List;

/** Categorised settings shell. Every displayed control maps to an applied runtime preference. */
public final class SettingsActivity extends Activity {
    public static final String PREFS = "lsx4_settings";
    private static final String EXTRA_SCREEN = "settings_screen";
    private static final int REQUEST_IMPORT_NGS2_MODULE = 4301;
    private static final String NGS2_MODULE_NAME = "libSceNgs2.sprx";
    private static final long MAX_SYSTEM_MODULE_BYTES = 128L * 1024L * 1024L;

    public static final String K_RES_MODE = "res_mode";
    public static final String K_GPU_BACKEND = "gpu_backend";
    public static final String K_AUDIO = "audio_enabled";
    public static final String K_OVERLAY = "gamepad_overlay";
    public static final String K_OVERLAY_OPACITY = "overlay_opacity";
    public static final String K_PERF_HUD = "perf_hud";
    public static final String K_KEEP_AWAKE = "keep_awake";
    public static final String K_AUTO_LAUNCH_STORE = "auto_launch_store";
    public static final String K_PERSISTENT_JIT_CACHE = "persistent_jit_cache";
    private static final String K_LEGACY_PERSISTENT_JIT_CACHE =
            "jit_persistent_jit_cache";
    public static final String K_DISABLE_DYNAMIC_SHADOWS = "disable_dynamic_shadows";
    public static final String K_DISABLE_SSAO = "disable_ssao";
    public static final String K_DISABLE_MOTION_BLUR = "disable_motion_blur";
    public static final String K_DISABLE_DEPTH_OF_FIELD = "disable_depth_of_field";
    public static final String K_DISABLE_ANTI_ALIASING = "disable_anti_aliasing";
    public static final String K_DISABLE_CHROMATIC_ABERRATION =
            "disable_chromatic_aberration";

    public static final int RES_NATIVE = 0;
    public static final int RES_1080P = 1;
    public static final int RES_720P = 2;
    public static final int RES_1620P = 3;
    public static final int RES_540P = 4;
    public static final int RES_360P = 5;
    public static final int RES_270P = 6;
    public static final int RES_180P = 7;

    private static final int[] RESOLUTION_MODE_ORDER = {
            RES_NATIVE, RES_1080P, RES_720P, RES_540P,
            RES_360P, RES_270P, RES_180P, RES_1620P
    };
    private static final int[] RESOLUTION_WIDTHS =
            {0, 1920, 1280, 2880, 960, 640, 480, 320};
    private static final int[] RESOLUTION_HEIGHTS =
            {0, 1080, 720, 1620, 540, 360, 270, 180};

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(AppLocale.wrap(base));
    }

    public static SharedPreferences prefs(Context context) {
        SharedPreferences preferences = context.getSharedPreferences(PREFS, MODE_PRIVATE);
        if (!preferences.contains(K_PERSISTENT_JIT_CACHE)
                && preferences.contains(K_LEGACY_PERSISTENT_JIT_CACHE)) {
            preferences.edit().putBoolean(K_PERSISTENT_JIT_CACHE,
                    preferences.getBoolean(K_LEGACY_PERSISTENT_JIT_CACHE, true)).apply();
        }
        return preferences;
    }

    private String[] resolutionModeLabels() {
        return new String[]{
                getString(R.string.resolution_native),
                "1080p (1920×1080)", "720p (1280×720)", "540p (960×540)",
                "360p (640×360)", "270p (480×270)", "180p (320×180)",
                getString(R.string.resolution_upscale_150)
        };
    }

    public static int resolutionModeToPosition(int mode) {
        for (int i = 0; i < RESOLUTION_MODE_ORDER.length; ++i) {
            if (RESOLUTION_MODE_ORDER[i] == mode) {
                return i;
            }
        }
        return resolutionModeToPosition(RES_NATIVE);
    }

    public static int resolutionModeAtPosition(int position) {
        return position >= 0 && position < RESOLUTION_MODE_ORDER.length
                ? RESOLUTION_MODE_ORDER[position] : RES_NATIVE;
    }

    public static int resolutionWidth(int mode) {
        return mode >= 0 && mode < RESOLUTION_WIDTHS.length
                ? RESOLUTION_WIDTHS[mode] : RESOLUTION_WIDTHS[RES_NATIVE];
    }

    public static int resolutionHeight(int mode) {
        return mode >= 0 && mode < RESOLUTION_HEIGHTS.length
                ? RESOLUTION_HEIGHTS[mode] : RESOLUTION_HEIGHTS[RES_NATIVE];
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        UiChrome.preparePortraitWindow(this);
        String screen = getIntent().getStringExtra(EXTRA_SCREEN);
        if (screen == null || screen.isEmpty()) {
            screen = "root";
        }
        setContentView(buildScreen(screen));
    }

    private View buildScreen(String screen) {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xff111417);
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        UiChrome.applyInsets(root, dp(20), dp(14), dp(20), dp(24));

        if ("root".equals(screen)) {
            buildRoot(root);
        } else {
            addScreenHeader(root, titleForScreen(screen));
            switch (screen) {
                case "graphics":
                    buildGraphics(root);
                    break;
                case "gpu":
                    buildGpu(root);
                    break;
                case "patches":
                    buildPatches(root);
                    break;
                case "audio":
                    buildAudio(root);
                    break;
                case "controls":
                    buildControls(root);
                    break;
                case "cache":
                    buildCache(root);
                    break;
                case "other":
                    buildOther(root);
                    break;
                default:
                    finish();
                    break;
            }
        }
        return scroll;
    }

    private void buildRoot(LinearLayout root) {
        root.addView(title(getString(R.string.settings_title)));
        root.addView(compactSection(getString(R.string.section_general)));

        Spinner language = spinner(AppLocale.NATIVE_LABELS);
        language.setSelection(AppLocale.selectedPosition(this), false);
        language.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position != AppLocale.selectedPosition(SettingsActivity.this)) {
                    AppLocale.select(SettingsActivity.this, position);
                    recreate();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
        root.addView(settingRow(getString(R.string.language), language),
                matchWrapWithMargins(0, 0, 0, dp(12)));

        root.addView(category(R.string.category_graphics, R.string.category_graphics_summary,
                "graphics"));
        root.addView(category(R.string.category_audio, R.string.category_audio_summary, "audio"));
        root.addView(category(R.string.category_controls, R.string.category_controls_summary,
                "controls"));
        root.addView(category(R.string.category_cache, R.string.category_cache_summary, "cache"));
        root.addView(category(R.string.category_other, R.string.category_other_summary, "other"));
    }

    private void buildGraphics(LinearLayout root) {
        root.addView(section(getString(R.string.graphics_resolution)));
        Spinner resolution = spinner(resolutionModeLabels());
        resolution.setSelection(resolutionModeToPosition(
                prefs(this).getInt(K_RES_MODE, RES_NATIVE)), false);
        resolution.setOnItemSelectedListener(saveSpinner(position ->
                prefs(this).edit().putInt(K_RES_MODE,
                        resolutionModeAtPosition(position)).apply()));
        root.addView(resolution, matchWrapWithMargins(0, dp(6), 0, dp(14)));

        root.addView(category(R.string.graphics_gpu, R.string.graphics_gpu_summary, "gpu"));
        root.addView(category(R.string.graphics_patches, R.string.graphics_patches_summary,
                "patches"));
    }

    private void buildGpu(LinearLayout root) {
        boolean openGl = GraphicsBackend.OpenGlBackend.ID.equals(
                prefs(this).getString(K_GPU_BACKEND, GraphicsBackend.VulkanBackend.ID));
        if (openGl) {
            prefs(this).edit().putString(
                    K_GPU_BACKEND, GraphicsBackend.VulkanBackend.ID).apply();
        }
        CheckBox vulkan = check(getString(R.string.gpu_vulkan), true);
        CheckBox openGlControl = check(
                getString(R.string.gpu_opengl) + " · "
                        + getString(R.string.gpu_opengl_pending), false);
        openGlControl.setEnabled(false);
        openGlControl.setAlpha(0.52f);

        vulkan.setOnClickListener(view -> {
            vulkan.setChecked(true);
            openGlControl.setChecked(false);
            prefs(this).edit().putString(
                    K_GPU_BACKEND, GraphicsBackend.VulkanBackend.ID).apply();
        });

        root.addView(vulkan, matchWrapWithMargins(0, dp(8), 0, dp(6)));
        root.addView(openGlControl, matchWrapWithMargins(0, 0, 0, dp(12)));
    }

    private void buildPatches(LinearLayout root) {
        List<GamePatchCatalog.Game> games = GamePatchCatalog.load(this);
        if (games.isEmpty()) {
            root.addView(hint(getString(R.string.patch_no_games)));
            return;
        }
        root.addView(label(getString(R.string.patch_select_game), 13, 0xffaab4c0));
        String[] names = new String[games.size()];
        for (int i = 0; i < games.size(); ++i) {
            names[i] = games.get(i).name;
        }
        Spinner game = spinner(names);
        root.addView(game, matchWrapWithMargins(0, dp(6), 0, dp(12)));
        LinearLayout patches = new LinearLayout(this);
        patches.setOrientation(LinearLayout.VERTICAL);
        root.addView(patches, matchWrap());
        Runnable render = () -> renderPatches(patches,
                games.get(Math.max(0, game.getSelectedItemPosition())));
        game.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                render.run();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
        render.run();
    }

    private void renderPatches(LinearLayout container, GamePatchCatalog.Game game) {
        container.removeAllViews();
        SharedPreferences preferences = prefs(this);
        for (GamePatchCatalog.Patch patch : game.patches) {
            String title = patch.titleResource != 0
                    ? getString(patch.titleResource) : patch.id;
            boolean disabled = preferences.getBoolean(patch.preferenceKey, false);
            CheckBox control = check(title, patch.inverted ? !disabled : disabled);
            control.setOnCheckedChangeListener((button, checked) -> preferences.edit()
                    .putBoolean(patch.preferenceKey, patch.inverted ? !checked : checked)
                    .apply());
            container.addView(control, cardMargins());
        }
    }

    private void buildAudio(LinearLayout root) {
        root.addView(preferenceCheck(R.string.audio_enable, K_AUDIO, true));

        TextView moduleStatus = hint(getString(isNgs2ModuleInstalled()
                ? R.string.audio_ngs2_module_ready
                : R.string.audio_ngs2_module_missing));
        moduleStatus.setPadding(dp(2), dp(14), dp(2), dp(8));
        root.addView(moduleStatus);

        TextView importModule = label(
                getString(R.string.audio_import_ngs2_module), 15, 0xffe3e8ef);
        importModule.setGravity(Gravity.CENTER_VERTICAL);
        importModule.setMinHeight(dp(58));
        importModule.setPadding(dp(18), dp(12), dp(16), dp(12));
        importModule.setBackground(cardBackground(0xff1b2026, 0xff2d3640, 14));
        importModule.setClickable(true);
        importModule.setFocusable(true);
        importModule.setOnClickListener(view -> pickNgs2Module());
        root.addView(importModule, cardMargins());

        TextView legalHint = hint(getString(R.string.audio_ngs2_module_hint));
        legalHint.setPadding(dp(2), dp(8), dp(2), 0);
        root.addView(legalHint);
    }

    private void pickNgs2Module() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try {
            startActivityForResult(intent, REQUEST_IMPORT_NGS2_MODULE);
        } catch (ActivityNotFoundException error) {
            Toast.makeText(this, R.string.file_picker_unavailable, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_IMPORT_NGS2_MODULE
                || resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        Uri source = data.getData();
        Thread importer = new Thread(() -> {
            try {
                importNgs2Module(source);
                runOnUiThread(() -> {
                    Toast.makeText(this, R.string.audio_ngs2_import_success,
                            Toast.LENGTH_LONG).show();
                    recreate();
                });
            } catch (Exception error) {
                runOnUiThread(() -> Toast.makeText(this,
                        getString(R.string.audio_ngs2_import_failed,
                                error.getMessage() == null
                                        ? error.getClass().getSimpleName()
                                        : error.getMessage()),
                        Toast.LENGTH_LONG).show());
            }
        }, "lsx4-ngs2-import");
        importer.setDaemon(true);
        importer.start();
    }

    private boolean isNgs2ModuleInstalled() {
        File module = new File(new File(new File(getFilesDir(), "lsx4-home"),
                "sys_modules"), NGS2_MODULE_NAME);
        if (!module.isFile() || module.length() < 64) {
            return false;
        }
        try {
            return hasSupportedModuleMagic(module);
        } catch (IOException ignored) {
            return false;
        }
    }

    private void importNgs2Module(Uri source) throws IOException {
        String displayName = selectedFileName(source);
        if (displayName == null || !NGS2_MODULE_NAME.equalsIgnoreCase(displayName.trim())) {
            throw new IOException(getString(R.string.audio_ngs2_error_name, NGS2_MODULE_NAME));
        }
        File modules = new File(new File(getFilesDir(), "lsx4-home"), "sys_modules");
        if (!modules.isDirectory() && !modules.mkdirs()) {
            throw new IOException(getString(R.string.audio_ngs2_error_directory));
        }
        File temporary = new File(modules, "." + NGS2_MODULE_NAME + ".importing");
        File destination = new File(modules, NGS2_MODULE_NAME);
        long copied = 0;
        try (InputStream input = getContentResolver().openInputStream(source);
             FileOutputStream output = new FileOutputStream(temporary)) {
            if (input == null) {
                throw new IOException(getString(R.string.audio_ngs2_error_open));
            }
            byte[] buffer = new byte[256 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                if (count == 0) {
                    continue;
                }
                if (copied > MAX_SYSTEM_MODULE_BYTES - count) {
                    throw new IOException(getString(R.string.audio_ngs2_error_size));
                }
                output.write(buffer, 0, count);
                copied += count;
            }
            output.getFD().sync();
        } catch (IOException error) {
            //noinspection ResultOfMethodCallIgnored
            temporary.delete();
            throw error;
        }
        if (copied < 64 || !hasSupportedModuleMagic(temporary)) {
            //noinspection ResultOfMethodCallIgnored
            temporary.delete();
            throw new IOException(getString(R.string.audio_ngs2_error_format));
        }
        try {
            Files.move(temporary.toPath(), destination.toPath(),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(temporary.toPath(), destination.toPath(),
                    StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private String selectedFileName(Uri source) {
        try (Cursor cursor = getContentResolver().query(source,
                new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) {
                return cursor.getString(0);
            }
        } catch (RuntimeException ignored) {
        }
        return source.getLastPathSegment();
    }

    private static boolean hasSupportedModuleMagic(File module) throws IOException {
        byte[] magic = new byte[4];
        try (FileInputStream input = new FileInputStream(module)) {
            if (input.read(magic) != magic.length) {
                return false;
            }
        }
        boolean elf = (magic[0] & 0xff) == 0x7f
                && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F';
        boolean ps4Self = (magic[0] & 0xff) == 0x4f
                && (magic[1] & 0xff) == 0x15
                && (magic[2] & 0xff) == 0x3d
                && (magic[3] & 0xff) == 0x1d;
        return elf || ps4Self;
    }

    private void buildControls(LinearLayout root) {
        root.addView(preferenceCheck(R.string.controls_enable, K_OVERLAY, true));
        root.addView(section(getString(R.string.controls_opacity)));
        TextView value = label("", 12, 0xff8d98a5);
        SeekBar opacity = new SeekBar(this);
        opacity.setMax(100);
        opacity.setMin(20);
        opacity.setProgress(prefs(this).getInt(K_OVERLAY_OPACITY, 74));
        value.setText(opacity.getProgress() + "%");
        opacity.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                value.setText(progress + "%");
                if (fromUser) {
                    prefs(SettingsActivity.this).edit()
                            .putInt(K_OVERLAY_OPACITY, progress).apply();
                }
            }

            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        root.addView(opacity, matchWrap());
        root.addView(value, matchWrap());
        TextView gamepadHint = hint(getString(R.string.controls_gamepad_hint));
        gamepadHint.setPadding(0, dp(12), 0, 0);
        root.addView(gamepadHint);
    }

    private void buildCache(LinearLayout root) {
        root.addView(preferenceCheck(R.string.cache_enable, K_PERSISTENT_JIT_CACHE, true));
        root.addView(hint(getString(R.string.cache_hint)));
    }

    private void buildOther(LinearLayout root) {
        root.addView(preferenceCheck(R.string.debug_hud, K_PERF_HUD, false));
    }

    private void addScreenHeader(LinearLayout root, String title) {
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        TextView back = label("‹", 38, 0xff6fb4ff);
        back.setContentDescription(getString(R.string.back));
        back.setGravity(Gravity.CENTER);
        back.setOnClickListener(view -> finish());
        bar.addView(back, new LinearLayout.LayoutParams(dp(48), dp(48)));
        bar.addView(title(title), new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(bar, matchWrapWithMargins(-dp(8), 0, 0, dp(10)));
    }

    private String titleForScreen(String screen) {
        switch (screen) {
            case "graphics": return getString(R.string.category_graphics);
            case "gpu": return getString(R.string.graphics_gpu);
            case "patches": return getString(R.string.graphics_patches);
            case "audio": return getString(R.string.category_audio);
            case "controls": return getString(R.string.category_controls);
            case "cache": return getString(R.string.category_cache);
            case "other": return getString(R.string.category_other);
            default: return getString(R.string.settings_title);
        }
    }

    private View category(int titleResource, int summaryResource, String screen) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.HORIZONTAL);
        card.setGravity(Gravity.CENTER_VERTICAL);
        card.setPadding(dp(16), dp(14), dp(12), dp(14));
        card.setBackground(cardBackground(0xff1b2026, 0xff2d3640, 16));
        card.setClickable(true);
        card.setFocusable(true);

        LinearLayout text = new LinearLayout(this);
        text.setOrientation(LinearLayout.VERTICAL);
        text.addView(label(getString(titleResource), 16, 0xffeef2f6));
        text.addView(label(getString(summaryResource), 12, 0xff8d98a5));
        card.addView(text, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        card.addView(label("›", 30, 0xff6fb4ff));
        card.setOnClickListener(view -> {
            Intent intent = new Intent(this, SettingsActivity.class);
            intent.putExtra(EXTRA_SCREEN, screen);
            startActivity(intent);
        });
        card.setLayoutParams(cardMargins());
        card.setElevation(dp(2));
        return card;
    }

    private CheckBox preferenceCheck(int title, String key, boolean defaultValue) {
        CheckBox control = check(getString(title), prefs(this).getBoolean(key, defaultValue));
        control.setOnCheckedChangeListener((button, checked) ->
                prefs(this).edit().putBoolean(key, checked).apply());
        control.setLayoutParams(cardMargins());
        return control;
    }

    private CheckBox check(String text, boolean checked) {
        CheckBox control = new CheckBox(this);
        control.setText(text);
        control.setTextSize(15);
        control.setTextColor(0xffe3e8ef);
        control.setButtonDrawable(createCheckBoxButtonDrawable(this));
        control.setCompoundDrawablePadding(dp(10));
        control.setGravity(Gravity.CENTER_VERTICAL);
        control.setMinHeight(dp(60));
        control.setChecked(checked);
        control.setPadding(dp(18), dp(12), dp(16), dp(12));
        control.setBackground(cardBackground(0xff1b2026, 0xff2d3640, 14));
        return control;
    }

    public static Drawable createCheckBoxButtonDrawable(Context context) {
        int size = Math.round(24 * context.getResources().getDisplayMetrics().density);
        StateListDrawable states = new StateListDrawable();
        states.addState(new int[]{android.R.attr.state_checked},
                new CheckMarkDrawable(size, true));
        states.addState(new int[0], new CheckMarkDrawable(size, false));
        return states;
    }

    private TextView title(String text) {
        return label(text, 24, 0xfff4f7fa);
    }

    private TextView subtitle(String text) {
        TextView view = label(text, 12, 0xff7f8a97);
        view.setPadding(0, dp(2), 0, dp(6));
        return view;
    }

    private TextView section(String text) {
        TextView view = label(text, 13, 0xff6fb4ff);
        view.setPadding(0, dp(20), 0, dp(6));
        return view;
    }

    private TextView compactSection(String text) {
        TextView view = label(text, 13, 0xff6fb4ff);
        view.setPadding(0, dp(8), 0, dp(4));
        return view;
    }

    private View settingRow(String title, View control) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(14), dp(8), dp(10), dp(8));
        row.setBackground(cardBackground(0xff1b2026, 0xff2d3640, 14));
        row.addView(label(title, 14, 0xffd7dee7), new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        row.addView(control, new LinearLayout.LayoutParams(
                dp(174), ViewGroup.LayoutParams.WRAP_CONTENT));
        return row;
    }

    private TextView hint(String text) {
        TextView view = label(text, 12, 0xff7f8a97);
        view.setLineSpacing(0, 1.15f);
        return view;
    }

    private TextView label(String text, int sp, int color) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(sp);
        view.setTextColor(color);
        return view;
    }

    private Spinner spinner(String[] values) {
        Spinner spinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, values);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        return spinner;
    }

    private AdapterView.OnItemSelectedListener saveSpinner(PositionConsumer consumer) {
        return new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                consumer.accept(position);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        };
    }

    private GradientDrawable cardBackground(int fill, int stroke, int radiusDp) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(fill);
        drawable.setCornerRadius(dp(radiusDp));
        drawable.setStroke(dp(1), stroke);
        return drawable;
    }

    private LinearLayout.LayoutParams cardMargins() {
        return matchWrapWithMargins(0, dp(5), 0, dp(5));
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private LinearLayout.LayoutParams matchWrapWithMargins(int left, int top, int right, int bottom) {
        LinearLayout.LayoutParams params = matchWrap();
        params.setMargins(left, top, right, bottom);
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private interface PositionConsumer {
        void accept(int position);
    }

    private static final class CheckMarkDrawable extends Drawable {
        private final int size;
        private final boolean checked;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        CheckMarkDrawable(int size, boolean checked) {
            this.size = size;
            this.checked = checked;
        }

        @Override
        public void draw(Canvas canvas) {
            Rect bounds = getBounds();
            float inset = Math.max(1f, size * 0.08f);
            float radius = size * 0.24f;
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(checked ? 0xff4aa3ff : 0xff151a20);
            canvas.drawRoundRect(bounds.left + inset, bounds.top + inset,
                    bounds.right - inset, bounds.bottom - inset, radius, radius, paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(Math.max(2f, size * 0.09f));
            paint.setColor(checked ? 0xff91caff : 0xff66717e);
            canvas.drawRoundRect(bounds.left + inset, bounds.top + inset,
                    bounds.right - inset, bounds.bottom - inset, radius, radius, paint);
            if (checked) {
                paint.setStrokeWidth(Math.max(2.5f, size * 0.12f));
                paint.setStrokeCap(Paint.Cap.ROUND);
                paint.setStrokeJoin(Paint.Join.ROUND);
                paint.setColor(0xffffffff);
                float left = bounds.left;
                float top = bounds.top;
                canvas.drawLine(left + size * 0.27f, top + size * 0.52f,
                        left + size * 0.43f, top + size * 0.68f, paint);
                canvas.drawLine(left + size * 0.43f, top + size * 0.68f,
                        left + size * 0.75f, top + size * 0.34f, paint);
            }
        }

        @Override
        public void setAlpha(int alpha) {
            paint.setAlpha(alpha);
        }

        @Override
        public void setColorFilter(ColorFilter colorFilter) {
            paint.setColorFilter(colorFilter);
        }

        @Override
        public int getOpacity() {
            return PixelFormat.TRANSLUCENT;
        }

        @Override
        public int getIntrinsicWidth() {
            return size;
        }

        @Override
        public int getIntrinsicHeight() {
            return size;
        }
    }
}
