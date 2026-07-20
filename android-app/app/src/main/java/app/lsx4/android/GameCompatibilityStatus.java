package app.lsx4.android;

/** User-maintained compatibility classification for an installed title. */
enum GameCompatibilityStatus {
    UNCLASSIFIED,
    UNSUPPORTED,
    SUPPORTED;

    int librarySortPriority() {
        switch (this) {
        case SUPPORTED:
            return 0;
        case UNSUPPORTED:
            return 1;
        case UNCLASSIFIED:
        default:
            return 2;
        }
    }
}
