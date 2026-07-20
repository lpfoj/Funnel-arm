package app.lsx4.android;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.StatFs;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.Log;
import android.view.Gravity;
import android.view.HapticFeedbackConstants;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;

/**
 * Portrait launcher view shared by the Android entry activity.
 *
 * <p>The launcher deliberately treats the installed filesystem as its source of truth. Installed
 * titles are shown only when their real directories are present; no synthetic/fake game records are
 * created. Imported document trees are copied into the app sandbox because the native emulator
 * runtime consumes ordinary filesystem paths and cannot launch a {@code content://} URI.</p>
 */
public final class LauncherUi {
    private static final String TAG = "LSX4.Library";
    public static final int REQUEST_IMPORT_EXTRACTED_GAME = 4101;
    public static final int REQUEST_IMPORT_PKG = 4102;

    private static final int MENU_IMPORT_DIRECTORY = 5101;
    private static final int MENU_IMPORT_PKG = 5102;
    private static final String EXTRA_EMBEDDED_AARCH64_JIT_BACKEND =
            "embedded_aarch64_jit_backend";
    private static final Pattern TITLE_ID = Pattern.compile("[A-Z0-9]{9}");
    private static final int MAX_SFO_BYTES = 1024 * 1024;
    private static final int COPY_BUFFER_BYTES = 1024 * 1024;
    private static final int MAX_TREE_DEPTH = 64;
    private static final int MAX_IMPORT_FILES = 200_000;
    private static final int MAX_IMPORT_NODES = 200_000;
    private static final int MAX_DIRECTORY_CHILDREN = 20_000;
    private static final int MAX_DOCUMENT_NAME_LENGTH = 255;
    private static final int MAX_DOCUMENT_ID_LENGTH = 4096;
    private static final int MAX_TREE_URI_LENGTH = 8192;
    private static final long MAX_DOCUMENT_METADATA_CHARS = 8L * 1024L * 1024L;
    private static final long FREE_SPACE_RESERVE_BYTES = 512L * 1024L * 1024L;
    private static final Object IMPORT_LOCK = new Object();
    private static final AtomicBoolean IMPORT_ACTIVE = new AtomicBoolean();

    private final Activity activity;
    private final AtomicBoolean workRunning = new AtomicBoolean();
    private final GameCompatibilityRepository compatibilityRepository;
    private final Handler holdHandler;
    private final int touchSlop;
    private LinearLayout gameList;
    private TextView status;
    private ProgressBar progress;
    private ImageView removeAction;
    private GameEntry selectedGame;
    private List<GameEntry> latestGames = Collections.emptyList();
    private View rootView;
    private volatile boolean destroyed;

    public LauncherUi(Activity activity) {
        if (activity == null) {
            throw new IllegalArgumentException("activity == null");
        }
        this.activity = activity;
        compatibilityRepository = new GameCompatibilityRepository(activity);
        holdHandler = new Handler(Looper.getMainLooper());
        touchSlop = ViewConfiguration.get(activity).getScaledTouchSlop();
    }

    /** Builds the complete portrait launcher view. The caller owns setContentView(). */
    public View buildView() {
        activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xff111417);
        UiChrome.applyInsets(root, dp(16), dp(12), dp(16), dp(14));

        LinearLayout appBar = new LinearLayout(activity);
        appBar.setOrientation(LinearLayout.HORIZONTAL);
        appBar.setGravity(Gravity.CENTER_VERTICAL);
        appBar.setPadding(dp(14), dp(9), dp(6), dp(9));
        GradientDrawable appBarBackground = new GradientDrawable();
        appBarBackground.setColor(0xff1b2026);
        appBarBackground.setCornerRadius(dp(16));
        appBarBackground.setStroke(dp(1), 0xff2a313a);
        appBar.setBackground(appBarBackground);
        appBar.setElevation(dp(4));

        LinearLayout titleColumn = new LinearLayout(activity);
        titleColumn.setOrientation(LinearLayout.VERTICAL);
        titleColumn.addView(label("LSX4", 23, 0xffeef2f6), matchWrap());
        status = label(activity.getString(R.string.library_scanning), 12, 0xff9aa6b2);
        titleColumn.addView(status, matchWrap());
        appBar.addView(titleColumn, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        progress = new ProgressBar(activity);
        progress.setIndeterminate(true);
        progress.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressLp = new LinearLayout.LayoutParams(dp(24), dp(24));
        progressLp.setMargins(dp(8), 0, dp(6), 0);
        appBar.addView(progress, progressLp);

        removeAction = new ImageView(activity);
        removeAction.setImageResource(R.drawable.ic_delete);
        removeAction.setPadding(dp(12), dp(12), dp(12), dp(12));
        removeAction.setContentDescription(activity.getString(R.string.menu_remove_game));
        removeAction.setClickable(true);
        removeAction.setFocusable(true);
        removeAction.setVisibility(View.GONE);
        removeAction.setOnClickListener(view -> removeSelectedGame());
        appBar.addView(removeAction, new LinearLayout.LayoutParams(dp(48), dp(48)));

        TextView add = toolbarAction("+", activity.getString(R.string.menu_add));
        add.setOnClickListener(this::showAddMenu);
        appBar.addView(add, new LinearLayout.LayoutParams(dp(48), dp(48)));

        ImageView settings = new ImageView(activity);
        settings.setImageResource(R.drawable.ic_tune);
        settings.setPadding(dp(12), dp(12), dp(12), dp(12));
        settings.setContentDescription(activity.getString(R.string.menu_settings));
        settings.setClickable(true);
        settings.setFocusable(true);
        settings.setOnClickListener(view ->
                activity.startActivity(new Intent(activity, SettingsActivity.class)));
        appBar.addView(settings, new LinearLayout.LayoutParams(dp(48), dp(48)));
        root.addView(appBar, matchWrap());

        gameList = new LinearLayout(activity);
        gameList.setOrientation(LinearLayout.VERTICAL);
        ScrollView scroll = new ScrollView(activity);
        scroll.setFillViewport(true);
        scroll.addView(gameList);
        LinearLayout.LayoutParams scrollLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f);
        scrollLp.setMargins(0, dp(10), 0, 0);
        root.addView(scroll, scrollLp);

        rootView = root;
        refresh();
        return root;
    }

    private TextView toolbarAction(String glyph, String description) {
        TextView action = label(glyph, 30, 0xffeef2f6);
        action.setGravity(Gravity.CENTER);
        action.setContentDescription(description);
        action.setClickable(true);
        action.setFocusable(true);
        return action;
    }

    private void showAddMenu(View anchor) {
        PopupMenu popup = new PopupMenu(activity, anchor, Gravity.END);
        Menu menu = popup.getMenu();
        menu.add(Menu.NONE, MENU_IMPORT_DIRECTORY, 0,
                activity.getString(R.string.menu_import_directory));
        menu.add(Menu.NONE, MENU_IMPORT_PKG, 1,
                activity.getString(R.string.menu_import_pkg));
        popup.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == MENU_IMPORT_DIRECTORY) {
                pickExtractedGameTree();
                return true;
            }
            if (item.getItemId() == MENU_IMPORT_PKG) {
                pickPkg();
                return true;
            }
            return false;
        });
        popup.show();
    }

    /** MainActivity-facing name kept intentionally small so the launcher can remain a helper. */
    public View createView() {
        return buildView();
    }

    public View getView() {
        return rootView;
    }

    /** Detaches UI references; an already-running atomic import is allowed to finish safely. */
    public void destroy() {
        destroyed = true;
        holdHandler.removeCallbacksAndMessages(null);
        rootView = null;
        gameList = null;
        status = null;
        progress = null;
        removeAction = null;
    }

    /** Returns true when the result belongs to this launcher. */
    public boolean handleActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != REQUEST_IMPORT_EXTRACTED_GAME && requestCode != REQUEST_IMPORT_PKG) {
            return false;
        }
        if (resultCode != Activity.RESULT_OK || data == null || data.getData() == null) {
            setStatus(activity.getString(R.string.import_cancelled));
            return true;
        }

        Uri uri = data.getData();
        takeReadPermission(uri, data.getFlags());
        if (requestCode == REQUEST_IMPORT_EXTRACTED_GAME) {
            runJob(activity.getString(R.string.import_checking_directory), () -> {
                try {
                    return importExtractedTree(uri);
                } finally {
                    releaseReadPermission(uri);
                }
            });
        } else {
            runJob(activity.getString(R.string.import_copying_pkg), () -> {
                try {
                    return importPkg(uri);
                } finally {
                    releaseReadPermission(uri);
                }
            });
        }
        return true;
    }

    /** Rescans the two real install roots without constructing any placeholder titles. */
    public void refresh() {
        if (!workRunning.compareAndSet(false, true)) {
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        setBusy(true, activity.getString(R.string.library_scanning));
        Thread worker = new Thread(() -> {
            List<GameEntry> games;
            String message;
            try {
                synchronized (IMPORT_LOCK) {
                    games = scanInstalledGames();
                }
                message = games.isEmpty()
                        ? activity.getString(R.string.library_empty_status)
                        : activity.getString(R.string.library_found, games.size());
            } catch (Exception e) {
                games = Collections.emptyList();
                Log.e(TAG, activity.getString(R.string.scan_error)
                        + " [" + e.getClass().getSimpleName() + "]");
                message = activity.getString(R.string.scan_error);
            }
            workRunning.set(false);
            finishUiWork(games, message);
        }, "ps4-library-scan");
        worker.setDaemon(true);
        worker.start();
    }

    private void pickExtractedGameTree() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        try {
            activity.startActivityForResult(intent, REQUEST_IMPORT_EXTRACTED_GAME);
        } catch (ActivityNotFoundException e) {
            toast(activity.getString(R.string.folder_picker_unavailable));
        }
    }

    private void pickPkg() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        try {
            activity.startActivityForResult(intent, REQUEST_IMPORT_PKG);
        } catch (ActivityNotFoundException e) {
            toast(activity.getString(R.string.file_picker_unavailable));
        }
    }

    private void takeReadPermission(Uri uri, int returnedFlags) {
        int flags = returnedFlags & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        flags &= Intent.FLAG_GRANT_READ_URI_PERMISSION;
        if (flags == 0) {
            return;
        }
        try {
            activity.getContentResolver().takePersistableUriPermission(uri, flags);
        } catch (SecurityException | UnsupportedOperationException ignored) {
            // Some providers allow the immediate read but do not support persisted grants. The
            // import still proceeds while the transient result grant is alive.
        }
    }

    private void releaseReadPermission(Uri uri) {
        try {
            activity.getContentResolver().releasePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException | UnsupportedOperationException ignored) {
        }
    }

    private void runJob(String initialStatus, ImportJob job) {
        if (!workRunning.compareAndSet(false, true)) {
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        if (!IMPORT_ACTIVE.compareAndSet(false, true)) {
            workRunning.set(false);
            setStatus(activity.getString(R.string.import_already_running));
            return;
        }
        setBusy(true, initialStatus);
        Thread worker = new Thread(() -> {
            String message;
            List<GameEntry> games;
            try {
                synchronized (IMPORT_LOCK) {
                    try {
                        checkCancelled();
                        message = job.run();
                    } catch (Exception e) {
                        Log.e(TAG, activity.getString(R.string.import_error)
                                + " [" + e.getClass().getSimpleName() + "]");
                        message = activity.getString(R.string.import_error);
                    }
                    try {
                        games = scanInstalledGames();
                    } catch (Exception ignored) {
                        games = Collections.emptyList();
                    }
                }
            } finally {
                IMPORT_ACTIVE.set(false);
                workRunning.set(false);
            }
            finishUiWork(games, message);
        }, "ps4-game-import");
        worker.setDaemon(true);
        worker.start();
    }

    private String importExtractedTree(Uri treeUri) throws Exception {
        ContentResolver resolver = activity.getContentResolver();
        if (treeUri.toString().length() > MAX_TREE_URI_LENGTH) {
            throw new IOException("слишком длинный URI выбранной папки");
        }
        String rootId = DocumentsContract.getTreeDocumentId(treeUri);
        if (rootId == null || rootId.length() > MAX_DOCUMENT_ID_LENGTH) {
            throw new IOException("некорректный document ID корневой папки");
        }
        Uri rootDocument = DocumentsContract.buildDocumentUriUsingTree(treeUri, rootId);

        SafNode eboot = findDirectChild(treeUri, rootDocument, "eboot.bin", false);
        SafNode sceSys = findDirectChild(treeUri, rootDocument, "sce_sys", true);
        if (eboot == null || sceSys == null) {
            throw new IOException("выбранная папка должна содержать eboot.bin и sce_sys");
        }
        SafNode param = findDirectChild(treeUri, sceSys.uri, "param.sfo", false);
        if (param == null) {
            throw new IOException("в sce_sys отсутствует param.sfo");
        }
        if (eboot.size >= 0 && eboot.size < 4) {
            throw new IOException("eboot.bin пуст или обрезан");
        }

        byte[] sourceSfo = readUriLimited(resolver, param.uri, MAX_SFO_BYTES);
        String titleId = normalizeTitleId(PkgInstaller.sfoString(sourceSfo, "TITLE_ID"));
        if (!isValidTitleId(titleId)) {
            throw new IOException("некорректный TITLE_ID в param.sfo");
        }
        String title = cleanSfoString(PkgInstaller.sfoString(sourceSfo, "TITLE"));

        File home = homeDir();
        File appRoot = appInstallRoot();
        ensureDirectory(appRoot);
        File destination = checkedChild(appRoot, titleId);
        if (destination.exists() || Files.isSymbolicLink(destination.toPath())) {
            throw new IOException("игра " + titleId
                    + " уже установлена; автоматическое слияние отключено");
        }
        File stagingRoot = checkedChild(home, "import-staging");
        ensureDirectory(stagingRoot);
        File staging = checkedChild(stagingRoot,
                titleId + "-" + Long.toUnsignedString(System.nanoTime()));
        ensureDirectory(staging);

        CopyProgress copied = new CopyProgress(availableImportBudget(stagingRoot));
        boolean installed = false;
        try {
            copySafDirectory(treeUri, rootDocument, staging, staging, 0, copied,
                    new HashSet<>());

            File stagedEboot = checkedChild(staging, "eboot.bin");
            File stagedSfo = checkedChild(checkedChild(staging, "sce_sys"), "param.sfo");
            if (!isPlausibleEboot(stagedEboot, staging)
                    || !safeRegularFileWithin(stagedSfo, staging)) {
                throw new IOException("копия игры не прошла проверку структуры");
            }
            String copiedId = readSfoValue(stagedSfo, "TITLE_ID");
            if (!titleId.equals(copiedId)) {
                throw new IOException("TITLE_ID изменился во время копирования");
            }

            writeInstallManifest(staging, titleId, title, treeUri.toString(), true,
                    "saf-extracted-tree");
            installStagingTree(staging, destination, stagingRoot);
            installed = true;
        } finally {
            if (!installed && staging.exists()) {
                deleteTreeWithin(staging, stagingRoot);
            }
        }

        return activity.getString(R.string.import_directory_success,
                displayTitle(title, titleId), titleId, copied.files, formatBytes(copied.bytes));
    }

    private String importPkg(Uri uri) throws Exception {
        File home = homeDir();
        File inbox = checkedChild(home, "pkg-inbox");
        ensureDirectory(inbox);

        String displayName = queryDisplayName(uri);
        String safeName = safeFileName(displayName, "import.pkg");
        if (!safeName.toLowerCase(Locale.ROOT).endsWith(".pkg")) {
            safeName += ".pkg";
        }
        File destination = uniqueChild(inbox, safeName);
        long importBudget = availableImportBudget(inbox);
        long declaredSize = querySize(uri);
        if (declaredSize > importBudget) {
            throw new IOException("недостаточно свободного места для PKG (нужно "
                    + formatBytes(declaredSize) + ")");
        }
        long copied = copyUriToFile(uri, destination, importBudget);
        postStatus(activity.getString(R.string.import_pkg_copied, formatBytes(copied)));
        boolean registered = false;
        try {
            // PkgInstaller historically trusts TITLE_ID when constructing a child File. Preflight
            // the direct param.sfo first so an untrusted provider cannot use a forged TITLE_ID for
            // path traversal before the existing importer is invoked.
            PkgMetadata metadata = preflightPkg(destination);
            if (!isValidTitleId(metadata.titleId)) {
                throw new IOException("некорректный TITLE_ID в PKG");
            }

            File appRoot = appInstallRoot();
            ensureDirectory(appRoot);
            File finalDirectory = checkedChild(appRoot, metadata.titleId);
            if (finalDirectory.exists() || Files.isSymbolicLink(finalDirectory.toPath())) {
                throw new IOException("игра " + metadata.titleId
                        + " уже установлена; PKG не будет менять её файлы");
            }

            File metadataStagingRoot = checkedChild(home, "pkg-install-staging");
            ensureDirectory(metadataStagingRoot);
            File stagingHome = checkedChild(metadataStagingRoot,
                    metadata.titleId + "-" + Long.toUnsignedString(System.nanoTime()));
            ensureDirectory(stagingHome);
            try {
                PkgInstaller.Result result = PkgInstaller.install(destination, stagingHome);
                if (!result.ok) {
                    throw new IOException(result.message == null ? "PkgInstaller отклонил PKG"
                            : result.message);
                }
                if (!metadata.titleId.equals(result.titleId)) {
                    throw new IOException("PkgInstaller вернул другой TITLE_ID");
                }
                File stagedDirectory = checkedChild(checkedChild(checkedChild(
                        checkedChild(stagingHome, "runtime-fs"), "user"), "app"),
                        metadata.titleId);
                if (!stagedDirectory.isDirectory()
                        || !safeRegularFileWithin(new File(new File(stagedDirectory, "sce_sys"),
                                "param.sfo"), stagedDirectory)) {
                    throw new IOException("PkgInstaller не создал полный metadata staging");
                }
                installStagingTree(stagedDirectory, finalDirectory, metadataStagingRoot);
                registered = true;

                if (result.ebootExtracted) {
                    return activity.getString(R.string.import_pkg_registered,
                            displayTitle(result.title, result.titleId));
                }
                return activity.getString(R.string.import_pkg_metadata_only,
                        displayTitle(result.title, result.titleId));
            } finally {
                if (stagingHome.exists()) {
                    deleteTreeWithin(stagingHome, metadataStagingRoot);
                }
            }
        } finally {
            if (!registered && destination.exists()) {
                // A rejected package must not accumulate in the private inbox.
                //noinspection ResultOfMethodCallIgnored
                destination.delete();
            }
        }
    }

    private List<GameEntry> scanInstalledGames() throws IOException {
        File home = homeDir();
        File[] roots = new File[]{appInstallRoot(), checkedChild(home, "games")};
        List<GameEntry> merged = new ArrayList<>();
        for (int rootPriority = 0; rootPriority < roots.length; rootPriority++) {
            File[] children = roots[rootPriority].listFiles();
            if (children == null) {
                continue;
            }
            for (File directory : children) {
                if (!directory.isDirectory() || Files.isSymbolicLink(directory.toPath())) {
                    continue;
                }
                try {
                    ensureWithin(directory, roots[rootPriority]);
                } catch (IOException escapedRoot) {
                    continue;
                }
                GameEntry candidate = readGameEntry(directory, rootPriority, roots[rootPriority]);
                if (candidate == null) {
                    continue;
                }
                int duplicate = findDuplicate(merged, candidate);
                if (duplicate < 0) {
                    merged.add(candidate);
                } else if (prefer(candidate, merged.get(duplicate))) {
                    merged.set(duplicate, candidate);
                }
            }
        }

        merged.sort(Comparator
                .comparing((GameEntry entry) -> entry.title.toLowerCase(Locale.ROOT))
                .thenComparing(entry -> entry.titleId));
        return merged;
    }

    private GameEntry readGameEntry(File directory, int rootPriority, File scanRoot) {
        try {
            ensureWithin(directory, scanRoot);
            File eboot = new File(directory, "eboot.bin");
            File sceSys = new File(directory, "sce_sys");
            File sfo = new File(sceSys, "param.sfo");
            File manifest = new File(directory, "executor-installed.json");
            boolean hasEboot = isPlausibleEboot(eboot, directory);
            boolean hasSfo = safeRegularFileWithin(sfo, directory);
            boolean hasManifest = safeRegularFileWithin(manifest, directory);
            if (!hasEboot && !hasSfo && !hasManifest) {
                return null;
            }

            String titleId = hasSfo ? readSfoValue(sfo, "TITLE_ID") : null;
            if (!isValidTitleId(titleId) && isValidTitleId(directory.getName())) {
                titleId = directory.getName();
            }
            if (!isValidTitleId(titleId)) {
                return null;
            }
            String title = hasSfo ? readSfoValue(sfo, "TITLE") : null;
            if (title == null || title.isEmpty()) {
                title = titleId;
            }
            File icon = new File(sceSys, "icon0.png");
            return new GameEntry(directory, eboot,
                    safeRegularFileWithin(icon, directory) ? icon : null,
                    titleId, title, hasEboot, rootPriority);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static int findDuplicate(List<GameEntry> entries, GameEntry candidate) {
        for (int i = 0; i < entries.size(); i++) {
            GameEntry existing = entries.get(i);
            if (candidate.titleId.equals(existing.titleId)) {
                return i;
            }
        }
        return -1;
    }

    private static boolean prefer(GameEntry candidate, GameEntry existing) {
        if (candidate.launchable != existing.launchable) {
            return candidate.launchable;
        }
        if (candidate.rootPriority != existing.rootPriority) {
            return candidate.rootPriority < existing.rootPriority;
        }
        if ((candidate.icon != null) != (existing.icon != null)) {
            return candidate.icon != null;
        }
        return candidate.directory.lastModified() > existing.directory.lastModified();
    }

    private void renderGameList(List<GameEntry> games) {
        if (gameList == null) {
            return;
        }
        List<GameEntry> orderedGames = new ArrayList<>(games);
        orderedGames.sort(Comparator
                .comparingInt((GameEntry entry) -> compatibilityRepository
                        .statusOf(entry.titleId).librarySortPriority())
                .thenComparing(entry -> entry.title.toLowerCase(Locale.ROOT))
                .thenComparing(entry -> entry.titleId));
        latestGames = orderedGames;
        if (selectedGame != null && !orderedGames.contains(selectedGame)) {
            selectedGame = null;
            updateRemoveAction();
        }
        gameList.removeAllViews();
        if (orderedGames.isEmpty()) {
            TextView empty = label(activity.getString(R.string.library_empty),
                    14, 0xffc5cbd3);
            empty.setGravity(Gravity.CENTER);
            empty.setPadding(dp(22), dp(34), dp(22), dp(34));
            empty.setBackground(roundedBackground(0xff171b20, 0xff2a313a, 18));
            gameList.addView(empty, matchWrap());
            return;
        }
        for (GameEntry game : orderedGames) {
            gameList.addView(gameRow(game), matchWrapWithMargins(0, 0, 0, 10));
        }
    }

    private View gameRow(GameEntry game) {
        AccessibleGameRow row = new AccessibleGameRow(activity);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(12), dp(12), dp(12), dp(12));
        boolean selected = selectedGame == game;
        GameCompatibilityStatus compatibility = compatibilityRepository.statusOf(game.titleId);
        int fill = game.launchable ? 0xff1b2026 : 0xff171b20;
        int stroke = 0xff2a313a;
        if (compatibility == GameCompatibilityStatus.SUPPORTED) {
            fill = 0xff192820;
            stroke = 0xff397352;
        } else if (compatibility == GameCompatibilityStatus.UNSUPPORTED) {
            fill = 0xff2b201d;
            stroke = 0xff8c4f43;
        }
        row.setBackground(roundedBackground(
                selected ? 0xff203754 : fill,
                selected ? 0xff6fb4ff : stroke, 18));
        row.setElevation(dp(2));

        ImageView icon = new ImageView(activity);
        icon.setScaleType(ImageView.ScaleType.CENTER_CROP);
        LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(dp(82), dp(82));
        iconLp.setMargins(0, 0, dp(16), 0);
        row.addView(icon, iconLp);
        Bitmap bitmap = decodeIcon(game.icon, 192);
        if (bitmap != null) {
            icon.setImageBitmap(bitmap);
        } else {
            icon.setBackgroundColor(0xff2a313a);
        }

        LinearLayout textColumn = new LinearLayout(activity);
        textColumn.setOrientation(LinearLayout.VERTICAL);
        TextView title = label(game.title, 16,
                game.launchable ? 0xfff2f5f8 : 0xff9aa6b2);
        title.setTypeface(android.graphics.Typeface.DEFAULT,
                android.graphics.Typeface.BOLD);
        textColumn.addView(title);
        textColumn.addView(label(game.titleId, 12, 0xff7f8a97));
        TextView holdHint = label(activity.getString(R.string.game_hold_waiting),
                11, 0xff7f8a97);
        holdHint.setPadding(0, dp(5), 0, 0);
        textColumn.addView(holdHint);
        ProgressBar holdProgress = new ProgressBar(
                activity, null, android.R.attr.progressBarStyleHorizontal);
        holdProgress.setMax((int) GameHoldAction.MARK_SUPPORTED_MS);
        holdProgress.setProgress(0);
        holdProgress.setProgressTintList(
                android.content.res.ColorStateList.valueOf(0xff6fb4ff));
        holdProgress.setProgressBackgroundTintList(
                android.content.res.ColorStateList.valueOf(0xff303741));
        holdProgress.setVisibility(View.GONE);
        LinearLayout.LayoutParams holdProgressLp =
                new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(4));
        holdProgressLp.setMargins(0, dp(5), 0, 0);
        textColumn.addView(holdProgress, holdProgressLp);
        if (!game.launchable) {
            TextView metadata = label(activity.getString(R.string.metadata_only),
                    11, 0xffffad66);
            metadata.setPadding(0, dp(3), 0, 0);
            textColumn.addView(metadata);
        }
        row.addView(textColumn, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        if (compatibility != GameCompatibilityStatus.UNCLASSIFIED) {
            TextView compatibilityMark = label(
                    compatibility == GameCompatibilityStatus.SUPPORTED ? "\u2713" : "\u00d7",
                    20,
                    compatibility == GameCompatibilityStatus.SUPPORTED
                            ? 0xff67d391 : 0xffff806f);
            compatibilityMark.setGravity(Gravity.CENTER);
            compatibilityMark.setContentDescription(activity.getString(
                    compatibility == GameCompatibilityStatus.SUPPORTED
                            ? R.string.game_status_supported
                            : R.string.game_status_unsupported));
            LinearLayout.LayoutParams markLp =
                    new LinearLayout.LayoutParams(dp(34), dp(34));
            markLp.setMargins(dp(8), 0, 0, 0);
            row.addView(compatibilityMark, markLp);
        }

        row.setClickable(true);
        row.setFocusable(true);
        row.setOnClickListener(view -> {
            if (selectedGame != null) {
                selectedGame = game;
                updateRemoveAction();
                renderGameList(latestGames);
            } else if (game.launchable) {
                launch(game);
            }
        });
        int statusText = compatibility == GameCompatibilityStatus.SUPPORTED
                ? R.string.game_status_supported
                : compatibility == GameCompatibilityStatus.UNSUPPORTED
                ? R.string.game_status_unsupported
                : R.string.game_status_unclassified;
        row.setContentDescription(activity.getString(R.string.game_item_accessibility,
                game.title, game.titleId, activity.getString(statusText)));
        row.setOnTouchListener(new GameHoldTouchListener(game, holdHint, holdProgress));
        return row;
    }

    private void handleGameHoldAction(GameEntry game, GameHoldAction action) {
        if (action == GameHoldAction.TAP) {
            return;
        }
        if (action == GameHoldAction.SELECT_FOR_REMOVAL) {
            boolean selected = selectedGame != game;
            selectedGame = selected ? game : null;
            updateRemoveAction();
            toast(activity.getString(selected
                    ? R.string.game_selected_for_removal
                    : R.string.game_selection_cleared, game.title));
            renderGameList(latestGames);
            return;
        }

        GameCompatibilityStatus status = action == GameHoldAction.MARK_SUPPORTED
                ? GameCompatibilityStatus.SUPPORTED
                : GameCompatibilityStatus.UNSUPPORTED;
        compatibilityRepository.setStatus(game.titleId, status);
        selectedGame = null;
        updateRemoveAction();
        toast(activity.getString(status == GameCompatibilityStatus.SUPPORTED
                ? R.string.game_marked_supported
                : R.string.game_marked_unsupported, game.title));
        renderGameList(latestGames);
    }

    private final class GameHoldTouchListener implements View.OnTouchListener {
        private final GameEntry game;
        private final TextView holdHint;
        private final ProgressBar holdProgress;
        private View pressedView;
        private android.graphics.drawable.Drawable restingBackground;
        private long downAtMs;
        private float downX;
        private float downY;
        private boolean active;
        private GameHoldAction reachedAction = GameHoldAction.TAP;
        private int currentStageText = R.string.game_hold_waiting;

        private final Runnable elapsedTicker = new Runnable() {
            @Override
            public void run() {
                if (!active || pressedView == null) {
                    return;
                }
                long elapsedMs = Math.max(0L,
                        android.os.SystemClock.uptimeMillis() - downAtMs);
                holdProgress.setProgress((int) Math.min(
                        elapsedMs, GameHoldAction.MARK_SUPPORTED_MS));
                double seconds =
                        Math.min(elapsedMs, GameHoldAction.MARK_SUPPORTED_MS) / 1000.0;
                holdHint.setText(activity.getString(R.string.game_hold_elapsed,
                        seconds, activity.getString(currentStageText)));
                holdHandler.postDelayed(this, 100L);
            }
        };

        private final Runnable selectionCue = () -> showHoldStage(
                GameHoldAction.SELECT_FOR_REMOVAL, R.string.game_hold_release_remove, 0xff6fb4ff,
                0xff203754, 0xff6fb4ff, HapticFeedbackConstants.CLOCK_TICK);
        private final Runnable unsupportedCue = () -> showHoldStage(
                GameHoldAction.MARK_UNSUPPORTED, R.string.game_hold_release_unsupported, 0xffff806f,
                0xff2b201d, 0xffb85d50, HapticFeedbackConstants.CLOCK_TICK);
        private final Runnable supportedCue = () -> showHoldStage(
                GameHoldAction.MARK_SUPPORTED, R.string.game_hold_release_supported, 0xff67d391,
                0xff192820, 0xff4f9a6c, HapticFeedbackConstants.LONG_PRESS);

        GameHoldTouchListener(GameEntry game, TextView holdHint,
                              ProgressBar holdProgress) {
            this.game = game;
            this.holdHint = holdHint;
            this.holdProgress = holdProgress;
        }

        @Override
        public boolean onTouch(View view, MotionEvent event) {
            switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                // A fresh gesture always owns a fresh timer/callback set, even if a platform or
                // accessibility service starts it without delivering the previous ACTION_CANCEL.
                cancel();
                pressedView = view;
                restingBackground = view.getBackground();
                downAtMs = android.os.SystemClock.uptimeMillis();
                downX = event.getX();
                downY = event.getY();
                active = true;
                reachedAction = GameHoldAction.TAP;
                // The row must own a stationary gesture from its first event. Delaying this call
                // lets ScrollView synthesize ACTION_CANCEL from tiny touch jitter before any hold
                // threshold is reached. A deliberate drag relinquishes ownership below.
                if (view.getParent() != null) {
                    view.getParent().requestDisallowInterceptTouchEvent(true);
                }
                currentStageText = R.string.game_hold_waiting;
                holdHint.setTypeface(android.graphics.Typeface.DEFAULT,
                        android.graphics.Typeface.BOLD);
                holdHint.setTextColor(0xff9aa6b2);
                holdHint.setPadding(dp(8), dp(6), dp(8), dp(6));
                holdHint.setBackground(roundedBackground(0xff151a20, 0xff39424d, 10));
                holdHint.setVisibility(View.VISIBLE);
                holdProgress.setProgress(0);
                holdProgress.setProgressTintList(
                        android.content.res.ColorStateList.valueOf(0xff6fb4ff));
                holdProgress.setVisibility(View.VISIBLE);
                elapsedTicker.run();
                holdHandler.postDelayed(selectionCue,
                        GameHoldAction.SELECT_FOR_REMOVAL_MS);
                holdHandler.postDelayed(unsupportedCue,
                        GameHoldAction.MARK_UNSUPPORTED_MS);
                holdHandler.postDelayed(supportedCue,
                        GameHoldAction.MARK_SUPPORTED_MS);
                return true;
            case MotionEvent.ACTION_MOVE:
                long moveElapsedMs = android.os.SystemClock.uptimeMillis() - downAtMs;
                // A finger naturally drifts during an eight-second hold. Once the first action
                // threshold is reached the gesture is locked to this row; cancelling it after
                // that point made 4 s and 8 s indistinguishable in normal use.
                if (active && moveElapsedMs < GameHoldAction.SELECT_FOR_REMOVAL_MS
                        && (Math.abs(event.getX() - downX) > touchSlop * 5
                        || Math.abs(event.getY() - downY) > touchSlop * 5)) {
                    cancel();
                }
                return true;
            case MotionEvent.ACTION_UP:
                if (!active) {
                    cancel();
                    return true;
                }
                long durationMs = Math.max(0L,
                        android.os.SystemClock.uptimeMillis() - downAtMs);
                // The release duration is authoritative. Handler callbacks are only visual cues
                // and can run late on a busy launcher frame; using the last callback here made an
                // eight-second hold occasionally execute the four-second action.
                GameHoldAction action = GameHoldAction.fromDuration(durationMs);
                cancel();
                if (action == GameHoldAction.TAP) {
                    view.performClick();
                } else {
                    handleGameHoldAction(game, action);
                }
                return true;
            case MotionEvent.ACTION_CANCEL:
            case MotionEvent.ACTION_POINTER_DOWN:
                cancel();
                return true;
            default:
                return true;
            }
        }

        private void cue(int feedbackConstant) {
            if (active && pressedView != null) {
                pressedView.performHapticFeedback(feedbackConstant);
            }
        }

        private void showHoldStage(GameHoldAction action, int textResource,
                                   int textColor, int fillColor,
                                   int strokeColor, int feedbackConstant) {
            if (!active || pressedView == null) {
                return;
            }
            reachedAction = action;
            holdHint.setText(textResource);
            currentStageText = textResource;
            holdHint.setTextColor(textColor);
            holdHint.setVisibility(View.VISIBLE);
            holdProgress.setProgressTintList(
                    android.content.res.ColorStateList.valueOf(textColor));
            pressedView.setBackground(roundedBackground(fillColor, strokeColor, 18));
            cue(feedbackConstant);
        }

        private void cancel() {
            active = false;
            reachedAction = GameHoldAction.TAP;
            holdHandler.removeCallbacks(selectionCue);
            holdHandler.removeCallbacks(unsupportedCue);
            holdHandler.removeCallbacks(supportedCue);
            holdHandler.removeCallbacks(elapsedTicker);
            holdHint.setText(R.string.game_hold_waiting);
            holdHint.setTypeface(android.graphics.Typeface.DEFAULT,
                    android.graphics.Typeface.NORMAL);
            holdHint.setTextColor(0xff7f8a97);
            holdHint.setPadding(0, dp(5), 0, 0);
            holdHint.setBackground(null);
            holdHint.setVisibility(View.VISIBLE);
            holdProgress.setProgress(0);
            holdProgress.setVisibility(View.GONE);
            if (pressedView != null && restingBackground != null) {
                pressedView.setBackground(restingBackground);
            }
            if (pressedView != null && pressedView.getParent() != null) {
                pressedView.getParent().requestDisallowInterceptTouchEvent(false);
            }
            pressedView = null;
            restingBackground = null;
        }
    }

    /**
     * Keeps the custom hold gesture compatible with accessibility services that invoke a row
     * through {@link View#performClick()} instead of synthesizing touch input.
     */
    private static final class AccessibleGameRow extends LinearLayout {
        AccessibleGameRow(android.content.Context context) {
            super(context);
        }

        @Override
        public boolean performClick() {
            return super.performClick();
        }
    }

    private void updateRemoveAction() {
        if (removeAction != null) {
            removeAction.setVisibility(selectedGame == null ? View.GONE : View.VISIBLE);
        }
    }

    private void removeSelectedGame() {
        final GameEntry game = selectedGame;
        if (game == null) {
            return;
        }
        selectedGame = null;
        updateRemoveAction();
        if (!workRunning.compareAndSet(false, true)) {
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        if (!IMPORT_ACTIVE.compareAndSet(false, true)) {
            workRunning.set(false);
            setStatus(activity.getString(R.string.operation_wait));
            return;
        }
        setBusy(true, activity.getString(R.string.game_removing, game.title));
        Thread worker = new Thread(() -> {
            String message;
            List<GameEntry> games;
            try {
                synchronized (IMPORT_LOCK) {
                    try {
                        File boundary = game.rootPriority == 0
                                ? appInstallRoot() : checkedChild(homeDir(), "games");
                        ensureWithin(game.directory, boundary);
                        deleteTreeWithin(game.directory, boundary);
                        compatibilityRepository.clear(game.titleId);
                        message = activity.getString(R.string.game_removed, game.title);
                    } catch (Exception error) {
                        Log.e(TAG, "Could not remove " + game.titleId, error);
                        message = activity.getString(R.string.game_remove_failed, game.title);
                    }
                    try {
                        games = scanInstalledGames();
                    } catch (Exception ignored) {
                        games = Collections.emptyList();
                    }
                }
            } finally {
                IMPORT_ACTIVE.set(false);
                workRunning.set(false);
            }
            finishUiWork(games, message);
        }, "lsx4-game-remove");
        worker.setDaemon(true);
        worker.start();
    }

    private void launch(GameEntry game) {
        if (workRunning.get() || IMPORT_ACTIVE.get()) {
            toast(activity.getString(R.string.import_wait_before_launch));
            return;
        }
        Intent intent = new Intent(activity, GameActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        intent.putExtra("autoload_existing_runtime", true);
        intent.putExtra("autoinit_runtime", true);
        intent.putExtra("scan_game_path", game.eboot.getAbsolutePath());
        intent.putExtra("launch_game_path", game.eboot.getAbsolutePath());
        intent.putExtra("fullscreen_render", true);
        intent.putExtra(EXTRA_EMBEDDED_AARCH64_JIT_BACKEND, true);
        try {
            activity.startActivity(intent);
        } catch (ActivityNotFoundException e) {
            toast(activity.getString(R.string.game_activity_unavailable));
        }
    }

    private void copySafDirectory(Uri treeUri, Uri sourceDirectory, File destination,
                                  File stagingBoundary, int depth, CopyProgress copied,
                                  Set<String> visitedDocumentIds)
            throws Exception {
        checkCancelled();
        if (depth > MAX_TREE_DEPTH) {
            throw new IOException("слишком глубокое дерево папок");
        }
        String documentId = DocumentsContract.getDocumentId(sourceDirectory);
        if (!visitedDocumentIds.add(documentId)) {
            throw new IOException("провайдер вернул циклическое дерево папок");
        }
        ensureWithin(destination, stagingBoundary);
        ensureDirectory(destination);
        Set<String> childNames = new HashSet<>();
        int remainingNodes = MAX_IMPORT_NODES - copied.nodes;
        if (remainingNodes <= 0) {
            throw new IOException("слишком много объектов в импортируемой игре");
        }
        long remainingMetadataChars = MAX_DOCUMENT_METADATA_CHARS - copied.metadataChars;
        if (remainingMetadataChars <= 0) {
            throw new IOException("слишком много метаданных в дереве игры");
        }
        List<SafNode> children = listChildren(treeUri, sourceDirectory,
                Math.min(MAX_DIRECTORY_CHILDREN, remainingNodes), remainingMetadataChars);
        if (children.size() > MAX_IMPORT_NODES - copied.nodes) {
            throw new IOException("слишком много объектов в импортируемой игре");
        }
        long reservedMetadataChars = 0;
        for (SafNode child : children) {
            reservedMetadataChars += child.metadataChars;
        }
        if (reservedMetadataChars > MAX_DOCUMENT_METADATA_CHARS - copied.metadataChars) {
            throw new IOException("слишком много метаданных в дереве игры");
        }
        copied.nodes += children.size();
        copied.metadataChars += reservedMetadataChars;
        for (SafNode child : children) {
            validateDocumentName(child.name);
            if (!childNames.add(child.name)) {
                throw new IOException("повторяющееся имя в папке: " + child.name);
            }
            File output = checkedChild(destination, child.name);
            ensureWithin(output, stagingBoundary);
            if (child.directory) {
                copySafDirectory(treeUri, child.uri, output, stagingBoundary, depth + 1, copied,
                        visitedDocumentIds);
            } else {
                if (child.size >= 0 && child.size > copied.maxBytes - copied.bytes) {
                    throw new IOException("недостаточно места для файла " + child.name);
                }
                copyDocument(child.uri, output, copied);
            }
        }
    }

    private void copyDocument(Uri source, File destination, CopyProgress copied) throws Exception {
        checkCancelled();
        if (copied.files >= MAX_IMPORT_FILES) {
            throw new IOException("слишком много файлов в импортируемой игре");
        }
        ContentResolver resolver = activity.getContentResolver();
        try (InputStream input = resolver.openInputStream(source);
             FileOutputStream output = new FileOutputStream(destination)) {
            if (input == null) {
                throw new IOException("провайдер не открыл " + source);
            }
            byte[] buffer = new byte[COPY_BUFFER_BYTES];
            int count;
            while ((count = input.read(buffer)) != -1) {
                checkCancelled();
                if (copied.bytes > copied.maxBytes - count) {
                    throw new IOException("импорт остановлен: заканчивается свободное место");
                }
                output.write(buffer, 0, count);
                copied.bytes += count;
                if (copied.bytes - copied.lastReportedBytes >= 32L * 1024L * 1024L) {
                    copied.lastReportedBytes = copied.bytes;
                    postStatus(activity.getString(R.string.import_copy_progress,
                            copied.files, formatBytes(copied.bytes)));
                }
            }
            output.getFD().sync();
        }
        copied.files++;
    }

    private List<SafNode> listChildren(Uri treeUri, Uri parentDocument) throws IOException {
        return listChildren(treeUri, parentDocument, MAX_DIRECTORY_CHILDREN,
                MAX_DOCUMENT_METADATA_CHARS);
    }

    private List<SafNode> listChildren(Uri treeUri, Uri parentDocument, int maximumChildren,
                                       long maximumMetadataChars) throws IOException {
        if (maximumChildren <= 0) {
            throw new IOException("слишком много объектов в импортируемой игре");
        }
        if (maximumMetadataChars <= 0) {
            throw new IOException("слишком много метаданных в дереве игры");
        }
        String parentId = DocumentsContract.getDocumentId(parentDocument);
        Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId);
        String[] projection = new String[]{
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
                DocumentsContract.Document.COLUMN_SIZE
        };
        List<SafNode> nodes = new ArrayList<>();
        long metadataChars = 0;
        try (Cursor cursor = activity.getContentResolver().query(
                childrenUri, projection, null, null, null)) {
            if (cursor == null) {
                throw new IOException("провайдер не вернул содержимое папки");
            }
            while (cursor.moveToNext()) {
                if (nodes.size() >= maximumChildren) {
                    throw new IOException("слишком много объектов в одной папке");
                }
                String documentId = cursor.getString(0);
                String name = cursor.getString(1);
                String mime = cursor.getString(2);
                long size = cursor.isNull(3) ? -1 : cursor.getLong(3);
                if (documentId == null || documentId.length() > MAX_DOCUMENT_ID_LENGTH) {
                    throw new IOException("провайдер вернул некорректный document ID");
                }
                if (name == null || name.length() > MAX_DOCUMENT_NAME_LENGTH) {
                    throw new IOException("слишком длинное имя в выбранной папке");
                }
                Uri uri = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                int nodeMetadataChars = documentId.length() + name.length()
                        + (mime == null ? 0 : mime.length()) + uri.toString().length() + 64;
                if (metadataChars > maximumMetadataChars - nodeMetadataChars) {
                    throw new IOException("слишком много метаданных в папке");
                }
                metadataChars += nodeMetadataChars;
                nodes.add(new SafNode(uri, name,
                        DocumentsContract.Document.MIME_TYPE_DIR.equals(mime), size,
                        nodeMetadataChars));
            }
        } catch (SecurityException e) {
            throw new IOException("нет доступа к выбранной папке", e);
        }
        return nodes;
    }

    private SafNode findDirectChild(Uri treeUri, Uri parent, String name, boolean directory)
            throws IOException {
        for (SafNode child : listChildren(treeUri, parent)) {
            if (name.equals(child.name) && child.directory == directory) {
                return child;
            }
        }
        return null;
    }

    private void installStagingTree(File staging, File destination, File stagingRoot)
            throws Exception {
        ensureWithin(staging, stagingRoot);
        ensureWithin(destination, appInstallRoot());
        if (destination.exists() || Files.isSymbolicLink(destination.toPath())) {
            throw new IOException("игра " + destination.getName()
                    + " уже установлена; слияние файлов запрещено");
        }
        try {
            Files.move(staging.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(staging.toPath(), destination.toPath());
        }
    }

    private static void moveReplace(File source, File target) throws IOException {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void deleteTreeWithin(File path, File boundary) throws IOException {
        ensureWithin(path, boundary);
        File[] children = path.listFiles();
        if (children != null) {
            for (File child : children) {
                if (child.isDirectory()) {
                    deleteTreeWithin(child, boundary);
                } else if (!child.delete()) {
                    throw new IOException("не удалось удалить временный файл " + child.getName());
                }
            }
        }
        if (path.exists() && !path.delete()) {
            throw new IOException("не удалось очистить staging " + path.getName());
        }
    }

    private void writeInstallManifest(File directory, String titleId, String title, String source,
                                      boolean ebootExtracted, String installedBy) throws Exception {
        JSONObject json = new JSONObject();
        json.put("titleId", titleId);
        json.put("title", title == null ? "" : title);
        json.put("source", source);
        json.put("ebootExtracted", ebootExtracted);
        json.put("installedBy", installedBy);
        File target = checkedChild(directory, "executor-installed.json");
        File temporary = checkedChild(directory, ".executor-installed.json.tmp");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(json.toString().getBytes(StandardCharsets.UTF_8));
            output.getFD().sync();
        }
        moveReplace(temporary, target);
    }

    private PkgMetadata preflightPkg(File pkg) throws Exception {
        try (RandomAccessFile file = new RandomAccessFile(pkg, "r")) {
            long length = file.length();
            if (length < 32 || u32be(file, 0) != 0x7f434e54L) {
                throw new IOException("файл не является PS4 PKG");
            }
            long entryCount = u32be(file, 0x10);
            long tableOffset = u32be(file, 0x18);
            if (entryCount <= 0 || entryCount > 8192 || tableOffset < 0
                    || tableOffset > length || entryCount > (length - tableOffset) / 32L) {
                throw new IOException("повреждена таблица PKG");
            }
            byte[] paramSfo = null;
            int paramSfoEntries = 0;
            for (long index = 0; index < entryCount; index++) {
                long entry = tableOffset + index * 32L;
                long id = u32be(file, entry);
                if (id != PkgInstaller.ENTRY_PARAM_SFO) {
                    continue;
                }
                paramSfoEntries++;
                if (paramSfoEntries > 1) {
                    throw new IOException("PKG содержит несколько param.sfo");
                }
                long offset = u32be(file, entry + 16);
                long size = u32be(file, entry + 20);
                if (size <= 0 || size > MAX_SFO_BYTES || offset > length
                        || size > length - offset) {
                    throw new IOException("повреждён param.sfo в PKG");
                }
                paramSfo = new byte[(int) size];
                file.seek(offset);
                file.readFully(paramSfo);
            }
            if (paramSfo == null) {
                throw new IOException("PKG не содержит param.sfo");
            }
            String id = normalizeTitleId(PkgInstaller.sfoString(paramSfo, "TITLE_ID"));
            String title = cleanSfoString(PkgInstaller.sfoString(paramSfo, "TITLE"));
            return new PkgMetadata(id, title);
        }
    }

    private long copyUriToFile(Uri uri, File destination, long maximumBytes) throws Exception {
        long copied = 0;
        try (InputStream input = activity.getContentResolver().openInputStream(uri);
             FileOutputStream output = new FileOutputStream(destination)) {
            if (input == null) {
                throw new IOException("провайдер не открыл выбранный PKG");
            }
            byte[] buffer = new byte[COPY_BUFFER_BYTES];
            int count;
            long lastReport = 0;
            while ((count = input.read(buffer)) != -1) {
                checkCancelled();
                if (copied > maximumBytes - count) {
                    throw new IOException("импорт остановлен: заканчивается свободное место");
                }
                output.write(buffer, 0, count);
                copied += count;
                if (copied - lastReport >= 64L * 1024L * 1024L) {
                    lastReport = copied;
                    postStatus(activity.getString(R.string.import_pkg_copy_progress,
                            formatBytes(copied)));
                }
            }
            output.getFD().sync();
        } catch (Exception e) {
            if (destination.exists()) {
                // destination is a unique file beneath the controlled inbox.
                //noinspection ResultOfMethodCallIgnored
                destination.delete();
            }
            throw e;
        }
        return copied;
    }

    private String queryDisplayName(Uri uri) {
        try (Cursor cursor = activity.getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) {
                return cursor.getString(0);
            }
        } catch (Exception ignored) {
        }
        return uri.getLastPathSegment();
    }

    private long querySize(Uri uri) {
        try (Cursor cursor = activity.getContentResolver().query(
                uri, new String[]{OpenableColumns.SIZE}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) {
                return Math.max(0, cursor.getLong(0));
            }
        } catch (Exception ignored) {
        }
        return -1;
    }

    private static byte[] readUriLimited(ContentResolver resolver, Uri uri, int maximum)
            throws IOException {
        try (InputStream input = resolver.openInputStream(uri);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            if (input == null) {
                throw new IOException("провайдер не открыл param.sfo");
            }
            byte[] buffer = new byte[8192];
            int total = 0;
            int count;
            while ((count = input.read(buffer)) != -1) {
                total += count;
                if (total > maximum) {
                    throw new IOException("param.sfo слишком большой");
                }
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private String readSfoValue(File sfo, String key) throws IOException {
        long length = sfo.length();
        if (length <= 0 || length > MAX_SFO_BYTES) {
            return null;
        }
        byte[] data = new byte[(int) length];
        try (FileInputStream input = new FileInputStream(sfo)) {
            int offset = 0;
            while (offset < data.length) {
                int count = input.read(data, offset, data.length - offset);
                if (count < 0) {
                    throw new IOException("обрезанный param.sfo");
                }
                offset += count;
            }
        }
        String value = cleanSfoString(PkgInstaller.sfoString(data, key));
        return "TITLE_ID".equals(key) ? normalizeTitleId(value) : value;
    }

    private static long u32be(RandomAccessFile file, long offset) throws IOException {
        file.seek(offset);
        int b0 = file.read();
        int b1 = file.read();
        int b2 = file.read();
        int b3 = file.read();
        if ((b0 | b1 | b2 | b3) < 0) {
            throw new IOException("unexpected EOF at " + offset);
        }
        return ((long) b0 << 24) | ((long) b1 << 16) | ((long) b2 << 8) | b3;
    }

    private File homeDir() throws IOException {
        return checkedChild(activity.getFilesDir(), "lsx4-home");
    }

    private File appInstallRoot() throws IOException {
        File runtimeFs = checkedChild(homeDir(), "runtime-fs");
        File user = checkedChild(runtimeFs, "user");
        return checkedChild(user, "app");
    }

    private long availableImportBudget(File existingDirectory) throws IOException {
        ensureDirectory(existingDirectory);
        StatFs stat = new StatFs(existingDirectory.getAbsolutePath());
        long available = stat.getAvailableBytes();
        long budget = available - FREE_SPACE_RESERVE_BYTES;
        if (budget <= 0) {
            throw new IOException("недостаточно свободного места; требуется резерв не менее 512 MiB");
        }
        return budget;
    }

    private static File checkedChild(File parent, String name) throws IOException {
        validateDocumentName(name);
        File child = new File(parent, name);
        ensureWithin(child, parent);
        return child;
    }

    private static File uniqueChild(File parent, String requestedName) throws IOException {
        File candidate = checkedChild(parent, requestedName);
        if (candidate.createNewFile()) {
            return candidate;
        }
        int dot = requestedName.lastIndexOf('.');
        String stem = dot > 0 ? requestedName.substring(0, dot) : requestedName;
        String extension = dot > 0 ? requestedName.substring(dot) : "";
        for (int suffix = 1; suffix <= 10000; suffix++) {
            candidate = checkedChild(parent, stem + "-" + suffix + extension);
            if (candidate.createNewFile()) {
                return candidate;
            }
        }
        throw new IOException("слишком много файлов с одинаковым именем");
    }

    private static void validateDocumentName(String name) throws IOException {
        if (name == null || name.isEmpty() || ".".equals(name) || "..".equals(name)
                || name.indexOf('/') >= 0 || name.indexOf('\\') >= 0
                || name.indexOf('\0') >= 0) {
            throw new IOException("небезопасное имя файла");
        }
    }

    private static String safeFileName(String name, String fallback) {
        if (name == null || name.trim().isEmpty()) {
            return fallback;
        }
        String safe = name.replaceAll("[^A-Za-z0-9._ -]", "_").trim();
        while (safe.startsWith(".")) {
            safe = safe.substring(1);
        }
        return safe.isEmpty() || ".".equals(safe) || "..".equals(safe) ? fallback : safe;
    }

    private static void ensureWithin(File child, File parent) throws IOException {
        String childPath = child.getCanonicalPath();
        String parentPath = parent.getCanonicalPath();
        if (!childPath.equals(parentPath)
                && !childPath.startsWith(parentPath + File.separator)) {
            throw new IOException("путь выходит за границу импорта");
        }
    }

    private static void ensureDirectory(File directory) throws IOException {
        if (directory.isDirectory()) {
            return;
        }
        if (directory.exists() || !directory.mkdirs()) {
            throw new IOException("не удалось создать " + directory.getName());
        }
    }

    private static boolean safeRegularFileWithin(File file, File boundary) {
        try {
            if (!file.isFile() || Files.isSymbolicLink(file.toPath())) {
                return false;
            }
            ensureWithin(file, boundary);
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    private static boolean isPlausibleEboot(File file, File boundary) {
        if (!safeRegularFileWithin(file, boundary) || file.length() < 4) {
            return false;
        }
        byte[] magic = new byte[4];
        try (FileInputStream input = new FileInputStream(file)) {
            if (input.read(magic) != magic.length) {
                return false;
            }
        } catch (IOException ignored) {
            return false;
        }
        boolean elf = (magic[0] & 0xff) == 0x7f
                && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F';
        boolean ps4Self = (magic[0] & 0xff) == 0x4f
                && (magic[1] & 0xff) == 0x15
                && (magic[2] & 0xff) == 0x3d
                && (magic[3] & 0xff) == 0x1d;
        return elf || ps4Self;
    }

    private void checkCancelled() throws IOException {
        if (destroyed || Thread.currentThread().isInterrupted()) {
            throw new IOException("импорт отменён: Activity закрыта");
        }
    }

    private static boolean isValidTitleId(String titleId) {
        return titleId != null && TITLE_ID.matcher(titleId).matches();
    }

    private static String normalizeTitleId(String value) {
        String clean = cleanSfoString(value);
        return clean == null ? null : clean.toUpperCase(Locale.ROOT);
    }

    private static String cleanSfoString(String value) {
        if (value == null) {
            return null;
        }
        int nul = value.indexOf('\0');
        String clean = (nul >= 0 ? value.substring(0, nul) : value).trim();
        return clean.isEmpty() ? null : clean;
    }

    private static String displayTitle(String title, String titleId) {
        return title == null || title.isEmpty() ? titleId : title;
    }

    private static String formatBytes(long bytes) {
        if (bytes >= 1024L * 1024L * 1024L) {
            return String.format(Locale.US, "%.2f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
        }
        if (bytes >= 1024L * 1024L) {
            return String.format(Locale.US, "%.1f MiB", bytes / (1024.0 * 1024.0));
        }
        if (bytes >= 1024L) {
            return String.format(Locale.US, "%.1f KiB", bytes / 1024.0);
        }
        return bytes + " B";
    }

    private static String friendlyMessage(Throwable throwable) {
        String message = throwable.getMessage();
        return message == null || message.trim().isEmpty()
                ? throwable.getClass().getSimpleName() : message;
    }

    private static Bitmap decodeIcon(File icon, int requestedPixels) {
        if (icon == null || !icon.isFile()) {
            return null;
        }
        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(icon.getAbsolutePath(), bounds);
            int sample = 1;
            while (bounds.outWidth / sample > requestedPixels * 2
                    || bounds.outHeight / sample > requestedPixels * 2) {
                sample *= 2;
            }
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = sample;
            return BitmapFactory.decodeFile(icon.getAbsolutePath(), options);
        } catch (Throwable ignored) {
            return null;
        }
    }

    private void finishUiWork(List<GameEntry> games, String message) {
        postUi(() -> {
            renderGameList(games);
            setBusy(false, message);
        });
    }

    private void setBusy(boolean busy, String message) {
        if (progress != null) {
            progress.setVisibility(busy ? View.VISIBLE : View.GONE);
        }
        if (status != null) {
            status.setText(message);
        }
    }

    private void setStatus(String message) {
        postUi(() -> {
            if (status != null) {
                status.setText(message);
            }
        });
    }

    private void postStatus(String message) {
        setStatus(message);
    }

    private void postUi(Runnable runnable) {
        activity.runOnUiThread(() -> {
            if (destroyed || activity.isFinishing()
                    || (Build.VERSION.SDK_INT >= 17 && activity.isDestroyed())) {
                return;
            }
            runnable.run();
        });
    }

    private void toast(String message) {
        postUi(() -> Toast.makeText(activity, message, Toast.LENGTH_LONG).show());
    }

    private TextView label(String text, int sp, int color) {
        TextView label = new TextView(activity);
        label.setText(text);
        label.setTextSize(sp);
        label.setTextColor(color);
        return label;
    }

    private GradientDrawable roundedBackground(int fill, int stroke, int radiusDp) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(fill);
        drawable.setCornerRadius(dp(radiusDp));
        drawable.setStroke(dp(1), stroke);
        return drawable;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private LinearLayout.LayoutParams matchWrapWithMargins(int left, int top, int right,
                                                            int bottom) {
        LinearLayout.LayoutParams lp = matchWrap();
        lp.setMargins(dp(left), dp(top), dp(right), dp(bottom));
        return lp;
    }

    private int dp(int value) {
        return Math.round(value * activity.getResources().getDisplayMetrics().density);
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private interface ImportJob {
        String run() throws Exception;
    }

    private static final class GameEntry {
        final File directory;
        final File eboot;
        final File icon;
        final String titleId;
        final String title;
        final boolean launchable;
        final int rootPriority;

        GameEntry(File directory, File eboot, File icon, String titleId, String title,
                  boolean launchable, int rootPriority) {
            this.directory = directory;
            this.eboot = eboot;
            this.icon = icon;
            this.titleId = titleId;
            this.title = title;
            this.launchable = launchable;
            this.rootPriority = rootPriority;
        }
    }

    private static final class SafNode {
        final Uri uri;
        final String name;
        final boolean directory;
        final long size;
        final int metadataChars;

        SafNode(Uri uri, String name, boolean directory, long size, int metadataChars) {
            this.uri = uri;
            this.name = name;
            this.directory = directory;
            this.size = size;
            this.metadataChars = metadataChars;
        }
    }

    private static final class CopyProgress {
        int nodes;
        int files;
        long bytes;
        long lastReportedBytes;
        long metadataChars;
        final long maxBytes;

        CopyProgress(long maxBytes) {
            this.maxBytes = maxBytes;
        }
    }

    private static final class PkgMetadata {
        final String titleId;
        final String title;

        PkgMetadata(String titleId, String title) {
            this.titleId = titleId;
            this.title = title;
        }
    }
}
