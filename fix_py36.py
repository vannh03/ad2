#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tự động kiểm tra và vá toàn diện mã nguồn cho Python 3.6.9 trên Jetson Nano.
Có thể chạy độc lập:
    python3 fix_py36.py
"""
import os
import re

print("🔧 Đang kiểm tra và đồng bộ tương thích Python 3.6.9 cho toàn bộ dự án...")

count = 0
for root, _, files in os.walk("ultralytics"):
    for file in files:
        if file.endswith(".py"):
            fpath = os.path.join(root, file)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = re.sub(
                r'([ \t]+)if ([a-zA-Z_0-9]+) := ([^:\n]+):',
                r'\1\2 = \3\n\1if \2:',
                content
            )
            new_content = re.sub(
                r'@functools\.lru_cache\b(?!\()',
                r'@functools.lru_cache()',
                new_content
            )
            if new_content != content:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                count += 1

print(f"✅ Hoàn tất! Đã kiểm tra và đồng bộ {count} file bổ sung.")
print("🚀 Bạn có thể chạy ngay: python3 run.py")
