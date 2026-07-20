package app.lsx4.android;

/** Maps an uninterrupted game-item hold to exactly one launcher action. */
enum GameHoldAction {
    TAP,
    SELECT_FOR_REMOVAL,
    MARK_UNSUPPORTED,
    MARK_SUPPORTED;

    static final long SELECT_FOR_REMOVAL_MS = 2_000L;
    static final long MARK_UNSUPPORTED_MS = 4_000L;
    static final long MARK_SUPPORTED_MS = 8_000L;

    static GameHoldAction fromDuration(long durationMs) {
        if (durationMs >= MARK_SUPPORTED_MS) {
            return MARK_SUPPORTED;
        }
        if (durationMs >= MARK_UNSUPPORTED_MS) {
            return MARK_UNSUPPORTED;
        }
        if (durationMs >= SELECT_FOR_REMOVAL_MS) {
            return SELECT_FOR_REMOVAL;
        }
        return TAP;
    }
}
