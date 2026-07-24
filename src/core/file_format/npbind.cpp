// SPDX-FileCopyrightText: Copyright 2025-2026 shadLauncher4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "npbind.h"

bool NPBindFile::Load(const std::string& path) {
    Clear();

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return false;

    std::streamsize sz = f.tellg();
    if (sz <= 0)
        return false;

    f.seekg(0, std::ios::beg);
    std::vector<u8> buf(static_cast<size_t>(sz));
    if (!f.read(reinterpret_cast<char*>(buf.data()), sz))
        return false;

    const u64 size = buf.size();
    if (size < sizeof(NpBindHeader))
        return false;

    memcpy(&m_header, buf.data(), sizeof(NpBindHeader));
    if (m_header.magic != NPBIND_MAGIC)
        return false;

    size_t offset = sizeof(NpBindHeader);

    m_bodies.reserve(static_cast<size_t>(m_header.num_entries));

    const u64 body_padding = 0x98;

    for (u64 bi = 0; bi < m_header.num_entries; ++bi) {
        if (offset + 4 * 4 > size)
            return false;

        NPBindBody body;

        auto read_entry = [&](NPBindEntryRaw& e) -> bool {
            if (offset + 4 > size)
                return false;

            memcpy(&e.type, &buf[offset], 2);
            memcpy(&e.size, &buf[offset + 2], 2);
            offset += 4;

            if (offset + e.size > size)
                return false;

            e.data.assign(buf.begin() + offset, buf.begin() + offset + e.size);
            offset += e.size;
            return true;
        };

        if (!read_entry(body.npcommid))
            return false;
        if (!read_entry(body.trophy))
            return false;
        if (!read_entry(body.unk1))
            return false;
        if (!read_entry(body.unk2))
            return false;

        if (offset + body_padding <= size) {
            offset += body_padding;
        } else {
            offset = size;
        }

        m_bodies.push_back(std::move(body));
    }

    if (size >= 20) {
        memcpy(m_digest, &buf[size - 20], 20);
    } else {
        memset(m_digest, 0, 20);
    }

    return true;
}

std::vector<std::string> NPBindFile::GetNpCommIds() const {
    std::vector<std::string> npcommids;
    npcommids.reserve(m_bodies.size());

    for (const auto& body : m_bodies) {
        if (!body.npcommid.data.empty()) {
            std::string raw_string(reinterpret_cast<const char*>(body.npcommid.data.data()),
                                   body.npcommid.data.size());
            npcommids.push_back(raw_string);
        }
    }

    return npcommids;
}
