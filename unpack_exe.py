import pefile
import os
import zlib

exe = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\SkhClny3.exe"
out_dir = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main\extracted"
os.makedirs(out_dir, exist_ok=True)

pe = pefile.PE(exe)

# Find .aspack section
aspack = None
for s in pe.sections:
    if b'aspack' in s.Name.lower():
        aspack = s
        break

if aspack:
    print(f"=== ASPACK SECTION ===")
    print(f"  VA: {aspack.VirtualAddress:#x}")
    print(f"  RawSize: {aspack.SizeOfRawData:#x}")
    print(f"  VirtSize: {aspack.Misc_VirtualSize:#x}")
    
    aspack_data = aspack.get_data()
    print(f"  Data loaded: {len(aspack_data)} bytes")
    
    # ASPack decompression (LZMA-based)
    # Try to find the decompressor
    print(f"  First 16 bytes: {aspack_data[:16].hex()}")
    
    # Try zlib
    for offset in range(0, min(len(aspack_data), 1000)):
        try:
            decompressed = zlib.decompress(aspack_data[offset:])
            print(f"  zlib at offset {offset}: {len(decompressed)} bytes decompressed")
            # Save
            with open(os.path.join(out_dir, "aspack_decompressed.bin"), 'wb') as f:
                f.write(decompressed)
            break
        except (zlib.error, OSError):  # не zlib-поток с этого смещения
            continue

# Try to use ResourceHacker or 7z
print("\n=== TRYING 7Z EXTRACTION ===")
import subprocess
exe_dir = os.path.dirname(exe)
res_dir = os.path.join(exe_dir, "extracted")
os.makedirs(res_dir, exist_ok=True)

# Try 7z to extract
result = subprocess.run(
    ['7z', 'x', exe, f'-o{res_dir}', '-y'],
    capture_output=True, text=True, timeout=30
)
print(f"  7z exit code: {result.returncode}")
if result.stdout:
    print(f"  stdout: {result.stdout[:500]}")
if result.stderr:
    print(f"  stderr: {result.stderr[:500]}")

# List what 7z extracted
for root, dirs, files in os.walk(res_dir):
    for f in files:
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(f"  {os.path.relpath(fp, res_dir)}: {size} bytes")
