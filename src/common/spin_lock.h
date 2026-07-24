// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <mutex>

namespace Common {

class SpinLock {
public:
    SpinLock() = default;

    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    SpinLock(SpinLock&&) = delete;
    SpinLock& operator=(SpinLock&&) = delete;

    void lock();
    void unlock();
    [[nodiscard]] bool try_lock();

private:
    std::atomic_flag lck = ATOMIC_FLAG_INIT;
};

class ParkingMutex {
public:
    ParkingMutex() = default;

    ParkingMutex(const ParkingMutex&) = delete;
    ParkingMutex& operator=(const ParkingMutex&) = delete;
    ParkingMutex(ParkingMutex&&) = delete;
    ParkingMutex& operator=(ParkingMutex&&) = delete;

    void lock() {
        mutex.lock();
    }

    void unlock() {
        mutex.unlock();
    }

    [[nodiscard]] bool try_lock() {
        return mutex.try_lock();
    }

private:
    std::mutex mutex;
};

}
