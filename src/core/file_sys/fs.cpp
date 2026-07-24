// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/config.h"
#include "common/string_util.h"
#include "core/file_sys/devices/logger.h"
#include "core/file_sys/devices/nop_device.h"
#include "core/file_sys/fs.h"

namespace Core::FileSys {

bool MntPoints::ignore_game_patches = false;

std::string RemoveTrailingSlashes(const std::string& path) {
    std::string path_sanitized = path;
    while (path_sanitized.ends_with("/")) {
        path_sanitized.pop_back();
    }
    return path_sanitized;
}

void MntPoints::Mount(const std::filesystem::path& host_folder, const std::string& guest_folder,
                      bool read_only) {
    std::scoped_lock lock{m_mutex};
    const auto guest_folder_sanitized = RemoveTrailingSlashes(guest_folder);
    m_mnt_pairs.emplace_back(host_folder, guest_folder_sanitized, read_only);
}

void MntPoints::Unmount(const std::filesystem::path& host_folder, const std::string& guest_folder) {
    std::scoped_lock lock{m_mutex};
    const auto guest_folder_sanitized = RemoveTrailingSlashes(guest_folder);
    auto it = std::remove_if(m_mnt_pairs.begin(), m_mnt_pairs.end(), [&](const MntPair& pair) {
        return pair.mount == guest_folder_sanitized;
    });
    m_mnt_pairs.erase(it, m_mnt_pairs.end());
}

void MntPoints::UnmountAll() {
    std::scoped_lock lock{m_mutex};
    m_mnt_pairs.clear();
}

std::filesystem::path MntPoints::GetHostPath(std::string_view path, bool* is_read_only,
                                             bool force_base_path) {
    std::string corrected_path(path);
    size_t pos = corrected_path.find("//");
    while (pos != std::string::npos) {
        corrected_path.replace(pos, 2, "/");
        pos = corrected_path.find("//", pos + 1);
    }

    if (path.length() > 255)
        return "";

    const std::optional<MntPair> mount = GetMount(corrected_path);
    if (!mount) {
        return "";
    }

    if (is_read_only) {
        *is_read_only = mount->read_only;
    }

    const auto corrected_path_sanitized = RemoveTrailingSlashes(corrected_path);
    if (corrected_path_sanitized == mount->mount) {
        return mount->host_path;
    }

    const auto rel_path = std::string_view{corrected_path}.substr(mount->mount.size() + 1);
    std::filesystem::path host_path = mount->host_path / rel_path;
    std::filesystem::path patch_path = mount->host_path;
    patch_path += "-UPDATE";
    if (!std::filesystem::exists(patch_path)) {
        patch_path = mount->host_path;
        patch_path += "-patch";
    }
    patch_path /= rel_path;

    if ((corrected_path.starts_with("/app0") || corrected_path.starts_with("/hostapp")) &&
        !force_base_path && !ignore_game_patches && std::filesystem::exists(patch_path)) {
        return patch_path;
    }

    if (!NeedsCaseInsensitiveSearch) {
        return host_path;
    }

    const auto search = [&](const auto host_path) {
        std::scoped_lock lk{m_mutex};
        path_parts.clear();
        auto current_path = host_path;
        while (!std::filesystem::exists(current_path)) {
            if (auto it = path_cache.find(current_path); it != path_cache.end()) {
                current_path = it->second;
                break;
            }
            path_parts.emplace_back(current_path.filename());
            current_path = current_path.parent_path();
        }
        auto guest_path = current_path;
        while (!path_parts.empty()) {
            const auto part = path_parts.back();
            const auto add_match = [&](const auto& host_part) {
                current_path /= host_part;
                guest_path /= part;
                path_cache[guest_path] = current_path;
                path_parts.pop_back();
            };
            if (std::filesystem::exists(current_path / part)) {
                add_match(part);
                continue;
            }
            const auto part_low = Common::ToLower(part.string());
            bool found_match = false;
            std::error_code iterator_error;
            for (const auto& path :
                 std::filesystem::directory_iterator(current_path, iterator_error)) {
                const auto candidate = path.path().filename();
                const auto filename = Common::ToLower(candidate.string());
                if (filename != part_low) {
                    continue;
                }
                add_match(candidate);
                found_match = true;
                break;
            }
            if (iterator_error) {
                return std::optional<std::filesystem::path>({});
            }
            if (!found_match) {
                return std::optional<std::filesystem::path>({});
            }
        }
        return std::optional<std::filesystem::path>(current_path);
    };

    if (!force_base_path && !ignore_game_patches) {
        if (const auto path = search(patch_path)) {
            return *path;
        }
    }
    if (const auto path = search(host_path)) {
        return *path;
    }

    return host_path;
}

void MntPoints::IterateDirectory(std::string_view guest_directory,
                                 const IterateDirectoryCallback& callback) {
    const auto base_path = GetHostPath(guest_directory, nullptr, true);
    const auto patch_path = GetHostPath(guest_directory, nullptr, false);
    const auto apply_patch = base_path != patch_path && std::filesystem::exists(patch_path);

    callback(base_path / ".", false);
    callback(base_path / "..", false);

    if (std::filesystem::exists(base_path)) {
        std::error_code iterator_error;
        for (const auto& entry : std::filesystem::directory_iterator(base_path, iterator_error)) {
            if (apply_patch) {
                const auto patch_entry_path = patch_path / entry.path().filename();
                if (std::filesystem::exists(patch_entry_path)) {
                    callback(patch_entry_path, !std::filesystem::is_directory(patch_entry_path));
                    continue;
                }
            }
            callback(entry.path(), !entry.is_directory());
        }
    }

    if (apply_patch) {
        std::error_code iterator_error;
        for (const auto& entry : std::filesystem::directory_iterator(patch_path, iterator_error)) {
            const auto base_entry_path = base_path / entry.path().filename();
            if (!std::filesystem::exists(base_entry_path)) {
                callback(entry.path(), !entry.is_directory());
            }
        }
    }
}

int HandleTable::CreateHandle() {
    std::scoped_lock lock{m_mutex};

    auto* file = new File{};
    file->is_opened = false;

    int existingFilesNum = m_files.size();

    for (int index = 0; index < existingFilesNum; index++) {
        if (m_files.at(index) == nullptr) {
            m_files[index] = file;
            return index;
        }
    }

    m_files.push_back(file);
    return m_files.size() - 1;
}

void HandleTable::DeleteHandle(int d) {
    std::scoped_lock lock{m_mutex};
    delete m_files.at(d);
    m_files[d] = nullptr;
}

File* HandleTable::GetFile(int d) {
    std::scoped_lock lock{m_mutex};
    if (d < 0 || d >= m_files.size()) {
        return nullptr;
    }
    return m_files.at(d);
}

File* HandleTable::GetSocket(int d) {
    std::scoped_lock lock{m_mutex};
    if (d < 0 || d >= m_files.size()) {
        return nullptr;
    }
    auto file = m_files.at(d);
    if (!file) {
        return nullptr;
    }
    if (file->type != Core::FileSys::FileType::Socket) {
        return nullptr;
    }
    return file;
}

File* HandleTable::GetEpoll(int d) {
    std::scoped_lock lock{m_mutex};
    if (d < 0 || d >= m_files.size()) {
        return nullptr;
    }
    auto file = m_files.at(d);
    if (file->type != Core::FileSys::FileType::Epoll) {
        return nullptr;
    }
    return file;
}

File* HandleTable::GetResolver(int d) {
    std::scoped_lock lock{m_mutex};
    if (d < 0 || d >= m_files.size()) {
        return nullptr;
    }
    auto file = m_files.at(d);
    if (file->type != Core::FileSys::FileType::Resolver) {
        return nullptr;
    }
    return file;
}

File* HandleTable::GetFile(const std::filesystem::path& host_name) {
    for (auto* file : m_files) {
        if (file != nullptr && file->m_host_name == host_name) {
            return file;
        }
    }
    return nullptr;
}

void HandleTable::CreateStdHandles() {
    auto setup = [this](const char* path, auto* device) {
        int fd = CreateHandle();
        auto* file = GetFile(fd);
        file->is_opened = true;
        file->type = FileType::Device;
        file->m_guest_name = path;
        file->device =
            std::shared_ptr<Devices::BaseDevice>{reinterpret_cast<Devices::BaseDevice*>(device)};
    };
    setup("/dev/stdin", new Devices::Logger("stdin", false));
    setup("/dev/stdout", new Devices::Logger("stdout", false));
    setup("/dev/stderr", new Devices::Logger("stderr", true));
}

int HandleTable::GetFileDescriptor(File* file) {
    std::scoped_lock lock{m_mutex};
    auto it = std::find(m_files.begin(), m_files.end(), file);

    if (it != m_files.end()) {
        return std::distance(m_files.begin(), it);
    }
    return 0;
}

}
