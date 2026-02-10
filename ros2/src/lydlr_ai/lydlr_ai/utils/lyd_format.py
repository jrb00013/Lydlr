# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# lydlr_ai/utils/lyd_format.py
import struct
import zlib

def save_lyd(path, chunks, lpips_score, modality_mask, timestamp):
    with open(path, 'wb') as f:
        f.write(b'LYDF')                            # Magic bytes
        f.write(int(timestamp).to_bytes(8, 'little'))
        f.write(lpips_score.to_bytes(4, 'little', signed=False))
        f.write(modality_mask.to_bytes(1, 'little'))
        f.write(len(chunks).to_bytes(1, 'little'))  # number of progressive chunks

        for chunk in chunks:
            f.write(len(chunk).to_bytes(4, 'little'))
            f.write(chunk)

def load_lyd(filename):
    with open(filename, 'rb') as f:
        header = f.read(13)  # 8 + 1 + 4 bytes
        timestamp, modality_mask, quality = struct.unpack('dBf', header)
        compressed_body = f.read()
        latent_bytes = zlib.decompress(compressed_body)
        latent_tensor = torch.frombuffer(latent_bytes, dtype=torch.float32)
        return timestamp, modality_mask, quality, latent_tensor

def load_lyd_progressive(path, max_chunks=4):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'LYDF'

        timestamp = int.from_bytes(f.read(8), 'little')
        lpips_score = int.from_bytes(f.read(4), 'little') / 10000.0
        modality_mask = int.from_bytes(f.read(1), 'little')
        num_chunks = int.from_bytes(f.read(1), 'little')

        chunks = []
        for _ in range(min(max_chunks, num_chunks)):
            size = int.from_bytes(f.read(4), 'little')
            chunk = f.read(size)
            chunks.append(chunk)

        return chunks, lpips_score, modality_mask, timestamp
