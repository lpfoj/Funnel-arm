package app.lsx4.android;

import android.content.Context;
import android.content.SharedPreferences;

import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * Persists the user's title classifications.
 *
 * <p>Each title owns one status key. Updating one game therefore cannot rewrite another game's
 * classification, while a single title can never be both supported and unsupported. The two
 * aggregate sets below are read only as a migration source for older installations.</p>
 */
final class GameCompatibilityRepository {
    private static final String PREFS_NAME = "lsx4_game_classification";
    private static final String SUPPORTED_TITLE_IDS = "supported_title_ids";
    private static final String UNSUPPORTED_TITLE_IDS = "unsupported_title_ids";
    private static final String TITLE_STATUS_PREFIX = "title_status_";
    private static final String STATUS_SUPPORTED = "supported";
    private static final String STATUS_UNSUPPORTED = "unsupported";

    private final SharedPreferences preferences;

    GameCompatibilityRepository(Context context) {
        preferences = context.getApplicationContext()
                .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        migrateLegacySets();
    }

    synchronized GameCompatibilityStatus statusOf(String titleId) {
        String key = normalize(titleId);
        if (key.isEmpty()) {
            return GameCompatibilityStatus.UNCLASSIFIED;
        }
        String stored = preferences.getString(TITLE_STATUS_PREFIX + key, "");
        if (STATUS_SUPPORTED.equals(stored)) {
            return GameCompatibilityStatus.SUPPORTED;
        }
        if (STATUS_UNSUPPORTED.equals(stored)) {
            return GameCompatibilityStatus.UNSUPPORTED;
        }
        // Preserve classifications written by builds that used the two aggregate sets. New writes
        // migrate the individual title to its own key, so unrelated games are never rewritten.
        if (readSet(SUPPORTED_TITLE_IDS).contains(key)) {
            return GameCompatibilityStatus.SUPPORTED;
        }
        if (readSet(UNSUPPORTED_TITLE_IDS).contains(key)) {
            return GameCompatibilityStatus.UNSUPPORTED;
        }
        return GameCompatibilityStatus.UNCLASSIFIED;
    }

    synchronized void setStatus(String titleId, GameCompatibilityStatus status) {
        String key = normalize(titleId);
        if (key.isEmpty()) {
            return;
        }

        SharedPreferences.Editor editor = preferences.edit();
        if (status == GameCompatibilityStatus.SUPPORTED) {
            editor.putString(TITLE_STATUS_PREFIX + key, STATUS_SUPPORTED);
        } else if (status == GameCompatibilityStatus.UNSUPPORTED) {
            editor.putString(TITLE_STATUS_PREFIX + key, STATUS_UNSUPPORTED);
        } else {
            editor.remove(TITLE_STATUS_PREFIX + key);
        }

        // Remove only this title from the legacy aggregate sets while migrating it. Do not rewrite
        // any other title's per-game key.
        Set<String> supported = readSet(SUPPORTED_TITLE_IDS);
        Set<String> unsupported = readSet(UNSUPPORTED_TITLE_IDS);
        supported.remove(key);
        unsupported.remove(key);
        editor
                .putStringSet(SUPPORTED_TITLE_IDS, supported)
                .putStringSet(UNSUPPORTED_TITLE_IDS, unsupported)
                .apply();
    }

    synchronized void clear(String titleId) {
        setStatus(titleId, GameCompatibilityStatus.UNCLASSIFIED);
    }

    synchronized Set<String> titleIds(GameCompatibilityStatus status) {
        if (status != GameCompatibilityStatus.SUPPORTED
                && status != GameCompatibilityStatus.UNSUPPORTED) {
            return Collections.emptySet();
        }
        Set<String> result = readSet(status == GameCompatibilityStatus.SUPPORTED
                ? SUPPORTED_TITLE_IDS : UNSUPPORTED_TITLE_IDS);
        String expected = status == GameCompatibilityStatus.SUPPORTED
                ? STATUS_SUPPORTED : STATUS_UNSUPPORTED;
        for (String preferenceKey : preferences.getAll().keySet()) {
            if (preferenceKey.startsWith(TITLE_STATUS_PREFIX)
                    && expected.equals(preferences.getString(preferenceKey, ""))) {
                result.add(preferenceKey.substring(TITLE_STATUS_PREFIX.length()));
            }
        }
        return Collections.unmodifiableSet(result);
    }

    private Set<String> readSet(String key) {
        return new HashSet<>(preferences.getStringSet(key, Collections.emptySet()));
    }

    /**
     * Converts the former two-set representation to one key per title.
     *
     * <p>The old representation made it possible for a stale or interrupted write to leave a
     * title in both sets. Preserve the status that the old reader exposed in that case
     * (supported), then remove both aggregate keys. Every subsequent update is isolated to the
     * selected title.</p>
     */
    private synchronized void migrateLegacySets() {
        if (!preferences.contains(SUPPORTED_TITLE_IDS)
                && !preferences.contains(UNSUPPORTED_TITLE_IDS)) {
            return;
        }

        Set<String> supported = normalizedSet(readSet(SUPPORTED_TITLE_IDS));
        Set<String> unsupported = normalizedSet(readSet(UNSUPPORTED_TITLE_IDS));
        Set<String> titleIds = new HashSet<>(unsupported);
        titleIds.addAll(supported);

        SharedPreferences.Editor editor = preferences.edit();
        for (String titleId : titleIds) {
            String statusKey = TITLE_STATUS_PREFIX + titleId;
            if (preferences.contains(statusKey)) {
                continue;
            }
            editor.putString(statusKey,
                    supported.contains(titleId) ? STATUS_SUPPORTED : STATUS_UNSUPPORTED);
        }
        editor.remove(SUPPORTED_TITLE_IDS);
        editor.remove(UNSUPPORTED_TITLE_IDS);
        editor.apply();
    }

    private static Set<String> normalizedSet(Set<String> titleIds) {
        Set<String> normalized = new HashSet<>();
        for (String titleId : titleIds) {
            String key = normalize(titleId);
            if (!key.isEmpty()) {
                normalized.add(key);
            }
        }
        return normalized;
    }

    private static String normalize(String titleId) {
        return titleId == null ? "" : titleId.trim().toUpperCase(Locale.ROOT);
    }
}
