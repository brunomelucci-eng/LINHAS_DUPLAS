#!/usr/bin/env tuple
import os
import sys
import argparse
import logging
import hashlib
import zipfile
import tempfile
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auditoria")

try:
    import yaml
except ImportError:
    yaml = None

def load_config(config_path: Path) -> dict:
    """
    Loads configuration from YAML file. Falls back to default settings
    with basic string-based parsing if PyYAML is not installed.
    """
    defaults = {
        'max_zip_mb': 450,
        'sample_limits': {'train': 20, 'val': 12, 'test': 12},
        'large_file_mb': 200,
        'large_log_mb': 20,
        'include_extensions': ['.py', '.yaml', '.yml', '.json', '.csv', '.md', '.txt', '.log', '.png', '.jpg', '.pt', '.npz'],
        'exclude_directories': ['.git', '.venv', 'venv', '__pycache__', 'node_modules', 'cache', 'temp', 'tmp', '.pytest_cache', '.mypy_cache', '.vscode', '.idea'],
        'preferred_model_names': ['best', 'approved', 'final', 'current', 'latest'],
        'sensitive_patterns': ['.env', 'private_key', 'credentials', 'token', 'secret', 'password']
    }
    if not config_path.exists():
        logger.debug(f"Configuration file {config_path} not found. Using defaults.")
        return defaults

    if yaml is not None:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    # Merge config with defaults
                    for k, v in defaults.items():
                        if k not in cfg:
                            cfg[k] = v
                    return cfg
        except Exception as e:
            logger.warning(f"Error loading YAML via PyYAML: {e}. Falling back to default settings.")
            return defaults

    # Simple line-based fallback parser for YAML if PyYAML is not available
    cfg = defaults.copy()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            current_section = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if not val:
                        current_section = key
                        if current_section == 'sample_limits':
                            cfg['sample_limits'] = {}
                        elif current_section in ['include_extensions', 'exclude_directories', 'preferred_model_names', 'sensitive_patterns']:
                            cfg[current_section] = []
                    else:
                        if val.startswith('[') and val.endswith(']'):
                            items = [x.strip().strip("'\"") for x in val[1:-1].split(',')]
                            cfg[key] = items
                        elif val.isdigit():
                            cfg[key] = int(val)
                        else:
                            cfg[key] = val.strip("'\"")
                elif line.startswith('-') and current_section:
                    val = line[1:].strip().strip("'\"")
                    if current_section in cfg:
                        cfg[current_section].append(val)
        return cfg
    except Exception as e:
        logger.warning(f"Error parsing YAML config: {e}. Using defaults.")
        return defaults

def find_project_root(start_path: Path) -> Path:
    """
    Finds the root directory of the project by searching upwards
    for common marker files like .git, requirements.txt, pyproject.toml.
    """
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists() or (parent / 'pyproject.toml').exists() or (parent / 'requirements.txt').exists():
            return parent
    return current

def calculate_sha256(file_path: Path) -> str:
    """
    Computes the SHA-256 hash of a file in blocks to avoid consuming high RAM.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def truncate_log(log_path: Path, max_log_mb: float) -> Path:
    """
    If a log file exceeds max_log_mb, returns a path to a temporary
    file containing the first 1000 lines and the last 1000 lines of the log.
    Otherwise, returns the original log_path.
    """
    file_size_mb = log_path.stat().st_size / (1024 * 1024)
    if file_size_mb <= max_log_mb:
        return log_path

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if len(lines) <= 2000:
            return log_path

        truncated_lines = (
            lines[:1000] +
            [f"... [TRUNCATED {len(lines) - 2000} LINES DUE TO SIZE LIMIT OF {max_log_mb}MB] ...\n"] +
            lines[-1000:]
        )
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"truncated_{log_path.name}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.writelines(truncated_lines)
        return temp_file
    except Exception as e:
        logger.error(f"Failed to truncate log {log_path}: {e}")
        return log_path

def scan_run_ids(root: Path) -> Dict[str, Path]:
    """
    Scans for execution run directories under outputs, runs, experiments, etc.
    Excludes standard non-run folders.
    """
    run_candidates = {}
    search_dirs = ['runs', 'outputs', 'experiments', 'modelo/approved', 'modelo/rejected']
    non_run_names = {'checkpoints', 'dataset', 'logs', 'cache', 'tmp', 'temp'}
    for sd in search_dirs:
        sd_path = root / sd
        if sd_path.exists() and sd_path.is_dir():
            for child in sd_path.iterdir():
                if child.is_dir() and child.name not in non_run_names:
                    if not child.name.startswith('.'):
                        run_candidates[child.name] = child
    return run_candidates

def select_run_id(run_candidates: Dict[str, Path], run_id_arg: Optional[str], latest_arg: bool) -> Optional[str]:
    """
    Validates and resolves the run_id to use.
    """
    if not run_candidates:
        if run_id_arg:
            raise ValueError(f"Run ID '{run_id_arg}' was specified, but no run directories were found in the project.")
        return None

    if run_id_arg:
        if run_id_arg not in run_candidates:
            raise ValueError(f"Specified Run ID '{run_id_arg}' not found. Available runs: {list(run_candidates.keys())}")
        return run_id_arg

    if latest_arg or True: # Default behavior is finding the latest
        # Sort candidates by modification time of their directory
        sorted_candidates = sorted(
            run_candidates.items(),
            key=lambda item: item[1].stat().st_mtime,
            reverse=True
        )
        return sorted_candidates[0][0]

def is_path_excluded(path: Path, root: Path, config: dict, output_dir: Optional[Path]) -> Tuple[bool, str]:
    """
    Checks if a path should be excluded based on config patterns.
    Returns (is_excluded, reason).
    """
    # 1. Skip if it is inside the output directory (to avoid recursion)
    if output_dir and (path == output_dir or output_dir in path.parents):
        return True, "Output folder file"

    # 2. Check excluded directories
    for part in path.relative_to(root).parts[:-1]:
        if part in config['exclude_directories']:
            return True, f"Excluded directory: {part}"

    # 3. Check excluded filenames and sensitive patterns
    name_lower = path.name.lower()
    for pattern in config['sensitive_patterns']:
        if pattern in name_lower:
            return True, f"Contains sensitive pattern: {pattern}"

    # 4. Check for temp/lock files
    if path.suffix in ['.pyc', '.pyo', '.tmp', '.bak', '.lock']:
        return True, f"Temporary or cache extension: {path.suffix}"

    return False, ""

def scan_files(root: Path, config: dict, selected_run_id: Optional[str], output_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scans the project root, categorizing included files and recording ignored files.
    """
    included_files = []
    ignored_files = []

    # Identify directories partitioned by other run_ids
    run_candidates = scan_run_ids(root)
    ignored_run_dirs = {path for rid, path in run_candidates.items() if selected_run_id and rid != selected_run_id}

    # Setup dataset sampling structures
    # We will record dataset files for sampling later
    dataset_files_by_split = {'train': [], 'val': [], 'test': []}

    for dirpath, dirnames, filenames in os.walk(root):
        dir_p = Path(dirpath)
        
        # Fast prune of directories to traverse
        pruned_dirs = []
        for d in dirnames:
            dp = dir_p / d
            is_exc, _ = is_path_excluded(dp, root, config, output_dir)
            # Check if this directory belongs to another run_id
            is_other_run = any(dp == ir or ir in dp.parents for ir in ignored_run_dirs)
            if not is_exc and not is_other_run:
                pruned_dirs.append(d)
        dirnames[:] = pruned_dirs

        for f in filenames:
            file_p = dir_p / f
            
            # Exclude check
            is_exc, exc_reason = is_path_excluded(file_p, root, config, output_dir)
            if is_exc:
                ignored_files.append({
                    'path': file_p,
                    'size': file_p.stat().st_size if file_p.exists() else 0,
                    'reason': exc_reason
                })
                continue

            # Check if file belongs to another run_id
            is_other_run = any(file_p == ir or ir in file_p.parents for ir in ignored_run_dirs)
            if is_other_run:
                ignored_files.append({
                    'path': file_p,
                    'size': file_p.stat().st_size,
                    'reason': f"Belongs to another run_id (current: {selected_run_id})"
                })
                continue

            file_size = file_p.stat().st_size

            # Size limit check (Etapa 5)
            if file_size > config['max_zip_mb'] * 1024 * 1024:
                ignored_files.append({
                    'path': file_p,
                    'size': file_size,
                    'reason': f"Exceeds max zip size limit ({config['max_zip_mb']} MB)"
                })
                continue

            # Identify if it is a dataset split file (Etapa 4)
            # Find splits under outputs/dataset/* or dataset/*
            rel_parts = file_p.relative_to(root).parts
            is_dataset = False
            for split in ['train', 'val', 'test']:
                if split in rel_parts and ('dataset' in rel_parts or 'IMAGE' in rel_parts or 'TALHOES' in rel_parts):
                    dataset_files_by_split[split].append(file_p)
                    is_dataset = True
                    break
            
            if is_dataset:
                continue

            # Categorize the file
            category = categorize_file(file_p, config)
            if category:
                included_files.append({
                    'path': file_p,
                    'category': category,
                    'reason': f"Standard {category} match"
                })
            else:
                ignored_files.append({
                    'path': file_p,
                    'size': file_size,
                    'reason': "File format or path did not match target categories"
                })

    # Sample dataset files (Etapa 4)
    # Sample limits from config
    limits = config['sample_limits']
    for split, files in dataset_files_by_split.items():
        files.sort()  # Keep deterministic selection
        limit = limits.get(split, 12)
        
        # We need to perform image/label pairing if separate images are used.
        # If .npz files are used, they are self-contained (both image & label inside), so we just take up to limit.
        selected_split_files = []
        rejected_split_files = []
        
        # Check if the split contains npz files
        npz_files = [fp for fp in files if fp.suffix.lower() == '.npz']
        if npz_files:
            selected_split_files = npz_files[:limit]
            rejected_split_files = npz_files[limit:] + [fp for fp in files if fp.suffix.lower() != '.npz']
        else:
            # Traditional image/label pairing logic
            # Separate files into images and labels
            img_exts = {'.png', '.jpg', '.jpeg'}
            images = [fp for fp in files if fp.suffix.lower() in img_exts]
            labels = [fp for fp in files if fp.suffix.lower() not in img_exts]
            
            # Select images up to limit
            selected_images = images[:limit]
            rejected_images = images[limit:]
            
            # Add selected images
            selected_split_files.extend(selected_images)
            
            # Add corresponding labels for each selected image
            selected_image_stems = {img.stem for img in selected_images}
            
            paired_labels = []
            unpaired_labels = []
            for lbl in labels:
                if lbl.stem in selected_image_stems:
                    paired_labels.append(lbl)
                else:
                    unpaired_labels.append(lbl)
            
            selected_split_files.extend(paired_labels)
            rejected_split_files.extend(rejected_images + unpaired_labels)

        # Record dataset files
        for fp in selected_split_files:
            included_files.append({
                'path': fp,
                'category': '02_amostras_dados',
                'reason': f"Dataset sample for {split}"
            })
        for fp in rejected_split_files:
            ignored_files.append({
                'path': fp,
                'size': fp.stat().st_size,
                'reason': f"Dataset file exceeds {split} sample limit ({limit})"
            })

    return included_files, ignored_files

def categorize_file(path: Path, config: dict) -> Optional[str]:
    """
    Determines which package category a file belongs to.
    """
    ext = path.suffix.lower()
    name = path.name.lower()
    
    # 03_modelos (checkpoints and models)
    if ext in ['.pt', '.pth', '.h5', '.onnx', '.joblib', '.pkl']:
        # Ensure it matches one of preferred model names
        if any(pmn in name for pmn in config['preferred_model_names']):
            return '03_modelos'
        return None  # Ignore intermediate checkpoints

    # 04_logs_complementares
    if ext == '.log' or 'logs' in path.parts:
        if ext in config['include_extensions']:
            return '04_logs_complementares'
        return None

    # 01_codigo_configuracoes_relatorios
    # Code extensions
    code_exts = {'.py', '.ipynb', '.js', '.ts', '.java', '.cs', '.cpp', '.c', '.h', '.sh', '.bat', '.ps1'}
    if ext in code_exts:
        return '01_codigo_configuracoes_relatorios'

    # Config filenames
    cfg_names = {'requirements.txt', 'environment.yml', 'pyproject.toml', 'setup.py', 'setup.cfg', 'dockerfile', 'docker-compose.yml', '.env.example'}
    if name in cfg_names or ext in ['.yaml', '.yml', '.json', '.toml', '.ini', '.cfg']:
        return '01_codigo_configuracoes_relatorios'

    # Reports
    report_keywords = {'report', 'relatorio', 'summary', 'resumo', 'manifest', 'metrics', 'metricas', 'quality', 'validation', 'validacao', 'diagnostic', 'diagnostico', 'audit', 'auditoria', 'results', 'resultado', 'status'}
    if ext in ['.csv', '.json', '.md', '.html', '.txt']:
        if any(kw in name for kw in report_keywords):
            return '01_codigo_configuracoes_relatorios'
            
    # Visual graphs/results
    vis_keywords = {'curva', 'grafico', 'confusion', 'matriz', 'sample', 'pred', 'prediction', 'overlay', 'probability', 'loss', 'training', 'val'}
    if ext in ['.png', '.jpg', '.jpeg', '.svg']:
        if any(kw in name for kw in vis_keywords):
            return '01_codigo_configuracoes_relatorios'

    return None

def pack_zips(
    root: Path,
    included_files: List[Dict[str, Any]],
    ignored_files: List[Dict[str, Any]],
    selected_run_id: Optional[str],
    output_dir: Path,
    config: dict,
    dry_run: bool
) -> Dict[str, Any]:
    """
    Groups files by category, handles log truncation, splits large zips,
    adds manifests, and compresses files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Structure files by category
    category_files = {
        '01_codigo_configuracoes_relatorios': [],
        '02_amostras_dados': [],
        '03_modelos': [],
        '04_logs_complementares': []
    }
    
    for f in included_files:
        cat = f['category']
        if cat in category_files:
            category_files[cat].append(f)
            
    # Track final created packages
    created_packages = {}
    
    # Store dynamic files that are created during zipping (e.g. truncated logs)
    temp_files_created = []

    # Map categories to clean display names
    for cat, files in category_files.items():
        if not files:
            continue
            
        # Group into chunks to guarantee each chunk fits under max_zip_mb
        # We use a threshold of 90% of max_zip_mb to leave space for manifests and compression overhead
        max_chunk_bytes = int(config['max_zip_mb'] * 1024 * 1024 * 0.9)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for f in files:
            orig_path = f['path']
            size = orig_path.stat().st_size
            if current_size + size > max_chunk_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(f)
            current_size += size
        if current_chunk:
            chunks.append(current_chunk)

        # Create zip for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            # Formulate zip name
            if len(chunks) == 1:
                zip_filename = f"{cat}.zip"
            else:
                # E.g. 02a_amostras_dados.zip, 02b_amostras_dados.zip
                suffix = chr(ord('a') + chunk_idx)
                zip_filename = f"{cat[:2]}{suffix}_{cat[3:]}.zip"
                
            zip_path = output_dir / zip_filename
            created_packages[zip_filename] = {
                'path': zip_path,
                'files': [],
                'total_size': 0
            }
            
            if dry_run:
                # Just mock file inclusion in dry-run
                for f in chunk:
                    orig_path = f['path']
                    arc_name = str(orig_path.relative_to(root).as_posix())
                    created_packages[zip_filename]['files'].append({
                        'original_path': orig_path,
                        'arc_name': arc_name,
                        'size': orig_path.stat().st_size,
                        'sha256': "DRY_RUN_HASH_MOCK",
                        'reason': f['reason']
                    })
                    created_packages[zip_filename]['total_size'] += orig_path.stat().st_size
                continue

            # Write manifest files to include inside the ZIP
            manifest_temp_dir = Path(tempfile.mkdtemp())
            
            manifest_file = manifest_temp_dir / "MANIFESTO_AUDITORIA.md"
            incl_csv = manifest_temp_dir / "arquivos_incluidos.csv"
            ign_csv = manifest_temp_dir / "arquivos_ignorados.csv"
            checksum_csv = manifest_temp_dir / "checksums_sha256.csv"
            
            # Prepare files list to pack
            files_to_pack = []
            
            # Truncate logs if needed (Etapa 5)
            for f in chunk:
                orig_path = f['path']
                arc_name = str(orig_path.relative_to(root).as_posix())
                
                pack_path = orig_path
                is_truncated = False
                if cat == '04_logs_complementares' and orig_path.stat().st_size > config['large_log_mb'] * 1024 * 1024:
                    truncated_temp = truncate_log(orig_path, config['large_log_mb'])
                    if truncated_temp != orig_path:
                        pack_path = truncated_temp
                        temp_files_created.append(truncated_temp)
                        is_truncated = True
                        
                file_size = pack_path.stat().st_size
                sha256_hash = calculate_sha256(pack_path)
                
                files_to_pack.append({
                    'pack_path': pack_path,
                    'original_path': orig_path,
                    'arc_name': arc_name,
                    'size': file_size,
                    'sha256': sha256_hash,
                    'reason': f['reason'] + (" (Truncated)" if is_truncated else "")
                })

            # Create CSVs
            # 1. Included files
            with open(incl_csv, 'w', newline='', encoding='utf-8') as csvf:
                writer = csv.writer(csvf)
                writer.writerow(['zip', 'categoria', 'caminho_original', 'caminho_no_zip', 'tamanho_bytes', 'sha256', 'motivo_inclusao', 'run_id'])
                for ftp in files_to_pack:
                    writer.writerow([
                        zip_filename, cat,
                        str(ftp['original_path'].relative_to(root).as_posix()),
                        ftp['arc_name'], ftp['size'], ftp['sha256'], ftp['reason'],
                        selected_run_id or ""
                    ])
                    
            # 2. Ignored files
            with open(ign_csv, 'w', newline='', encoding='utf-8') as csvf:
                writer = csv.writer(csvf)
                writer.writerow(['caminho_original', 'tamanho_bytes', 'motivo_exclusao'])
                for ign in ignored_files:
                    writer.writerow([
                        str(ign['path'].relative_to(root).as_posix()),
                        ign['size'], ign['reason']
                    ])
                    
            # 3. Checksums
            with open(checksum_csv, 'w', newline='', encoding='utf-8') as csvf:
                writer = csv.writer(csvf)
                writer.writerow(['arquivo', 'sha256', 'tamanho_bytes'])
                for ftp in files_to_pack:
                    writer.writerow([ftp['arc_name'], ftp['sha256'], ftp['size']])
                    
            # 4. Write Markdown Manifesto
            with open(manifest_file, 'w', encoding='utf-8') as mf:
                mf.write(f"# Manifesto de Auditoria\n\n")
                mf.write(f"- **Data e Hora de Geração**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                mf.write(f"- **Raiz do Projeto**: `{root.as_posix()}`\n")
                mf.write(f"- **Execução (Run ID)**: `{selected_run_id or 'Sem Run ID (Flat layout)'}`\n")
                mf.write(f"- **Pacote ZIP**: `{zip_filename}`\n")
                mf.write(f"- **Quantidade de Arquivos**: {len(files_to_pack)}\n")
                mf.write(f"- **Tamanho Total**: {sum(ftp['size'] for ftp in files_to_pack)} bytes\n\n")
                mf.write(f"## Arquivos Incluídos neste Pacote\n\n")
                mf.write(f"| Caminho | Tamanho (Bytes) | Hash SHA-256 | Motivo |\n")
                mf.write(f"| --- | --- | --- | --- |\n")
                for ftp in files_to_pack:
                    mf.write(f"| `{ftp['arc_name']}` | {ftp['size']} | `{ftp['sha256']}` | {ftp['reason']} |\n")

            # Zip creation
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add Manifest files
                zf.write(manifest_file, "MANIFESTO_AUDITORIA.md")
                zf.write(incl_csv, "arquivos_incluidos.csv")
                zf.write(ign_csv, "arquivos_ignorados.csv")
                zf.write(checksum_csv, "checksums_sha256.csv")
                
                # Add actual files
                for ftp in files_to_pack:
                    zf.write(ftp['pack_path'], ftp['arc_name'])
                    created_packages[zip_filename]['files'].append(ftp)
                    
            created_packages[zip_filename]['total_size'] = zip_path.stat().st_size
            
            # Clean up temp manifest files
            for ftemp in [manifest_file, incl_csv, ign_csv, checksum_csv]:
                if ftemp.exists():
                    ftemp.unlink()
            manifest_temp_dir.rmdir()

    # Cleanup temp truncated log files
    for t_file in temp_files_created:
        if t_file.exists():
            t_file.unlink()

    return created_packages

def run_zip_integrity_checks(created_packages: Dict[str, Any], root: Path) -> List[str]:
    """
    Tests ZIP files for CRC corruption, extracts to temporary folders,
    re-verifies checksums against manifest details, and validates size limits.
    """
    failures = []
    for zip_name, pkg in created_packages.items():
        zip_path = pkg['path']
        logger.info(f"Running integrity checks for {zip_name}...")
        
        # 1. Size check
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if size_mb > 450:
            failures.append(f"{zip_name} size ({size_mb:.2f} MB) exceeds the 450 MB limit.")
            continue
            
        # 2. Open and test ZIP
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Test zip CRC
                corrupt_file = zf.testzip()
                if corrupt_file:
                    failures.append(f"{zip_name} contains corrupt file: {corrupt_file}")
                    continue
                    
                # Extract to temp directory to verify hashes
                with tempfile.TemporaryDirectory() as temp_dir_str:
                    temp_dir = Path(temp_dir_str)
                    zf.extractall(temp_dir)
                    
                    # Read checksums_sha256.csv from extracted content
                    checksum_file = temp_dir / "checksums_sha256.csv"
                    if not checksum_file.exists():
                        failures.append(f"{zip_name} is missing checksums_sha256.csv.")
                        continue
                        
                    checksums_dict = {}
                    with open(checksum_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            checksums_dict[row['arquivo']] = {
                                'sha256': row['sha256'],
                                'tamanho': int(row['tamanho_bytes'])
                            }
                            
                    # Verify each file
                    for arc_name, meta in checksums_dict.items():
                        extracted_file = temp_dir / arc_name
                        if not extracted_file.exists():
                            failures.append(f"{zip_name}: File {arc_name} listed in manifest but not found in ZIP.")
                            continue
                            
                        # Calculate SHA256
                        hash_calc = calculate_sha256(extracted_file)
                        if hash_calc != meta['sha256']:
                            failures.append(f"{zip_name}: File {arc_name} SHA-256 mismatch! Expected: {meta['sha256']}, got: {hash_calc}")
                            
                        # Check size
                        sz = extracted_file.stat().st_size
                        if sz != meta['tamanho']:
                            failures.append(f"{zip_name}: File {arc_name} size mismatch! Expected: {meta['tamanho']}, got: {sz}")
                            
        except Exception as e:
            failures.append(f"Failed to verify zip {zip_name} due to exception: {e}")
            
    return failures

def write_final_report(
    root: Path,
    created_packages: Dict[str, Any],
    ignored_files: List[Dict[str, Any]],
    selected_run_id: Optional[str],
    output_dir: Path,
    integrity_failures: List[str]
) -> Path:
    """
    Generates RELATORIO_FINAL_PACOTE.md in the output directory.
    """
    report_path = output_dir / "RELATORIO_FINAL_PACOTE.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Relatório de Auditoria Técnica de Pacotes ZIP\n\n")
        f.write(f"- **Data/Hora**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Diretório Raiz**: `{root.as_posix()}`\n")
        f.write(f"- **Execução Selecionada (Run ID)**: `{selected_run_id or 'Sem Run ID (Flat layout)'}`\n")
        f.write(f"- **Diretório de Saída**: `{output_dir.as_posix()}`\n\n")
        
        # Integrity checks summary
        f.write(f"## Status de Integridade dos ZIPs\n\n")
        if integrity_failures:
            f.write(f"> [!CAUTION]\n")
            f.write(f"> Falhas de integridade detectadas nos pacotes ZIP:\n")
            for fail in integrity_failures:
                f.write(f"> - {fail}\n")
            f.write(f"\n")
        else:
            f.write(f"> [!NOTE]\n")
            f.write(f"> Todos os pacotes ZIP passaram nos testes de CRC e validação de hashes SHA-256 com sucesso!\n\n")

        # ZIPs created summary
        f.write(f"## Pacotes ZIP Criados\n\n")
        f.write(f"| Nome do ZIP | Tamanho (MB) | Quantidade de Arquivos |\n")
        f.write(f"| --- | --- | --- |\n")
        total_files_packed = 0
        for name, pkg in created_packages.items():
            sz_mb = pkg['total_size'] / (1024 * 1024)
            fcnt = len(pkg['files'])
            total_files_packed += fcnt
            f.write(f"| `{name}` | {sz_mb:.2f} MB | {fcnt} |\n")
        f.write(f"\n")

        # Included files breakdown
        f.write(f"## Arquivos Incluídos por Categoria\n\n")
        for name, pkg in created_packages.items():
            f.write(f"### `{name}`\n")
            f.write(f"| Arquivo | Tamanho | SHA-256 | Motivo |\n")
            f.write(f"| --- | --- | --- | --- |\n")
            for file_info in pkg['files']:
                f.write(f"| `{file_info['arc_name']}` | {file_info['size']} bytes | `{file_info['sha256'][:10]}...` | {file_info['reason']} |\n")
            f.write(f"\n")

        # Excluded / Ignored files
        f.write(f"## Arquivos Ignorados / Grandes Excluídos\n\n")
        large_files = [x for x in ignored_files if "Exceeds max zip size limit" in x['reason']]
        f.write(f"**Total de arquivos ignorados**: {len(ignored_files)}\n\n")
        
        if large_files:
            f.write(f"### Arquivos Grandes Excluídos (>450MB):\n")
            f.write(f"| Arquivo | Tamanho (MB) | Justificativa |\n")
            f.write(f"| --- | --- | --- |\n")
            for lf in large_files:
                f.write(f"| `{lf['path'].relative_to(root).as_posix()}` | {lf['size'] / (1024 * 1024):.2f} MB | {lf['reason']} |\n")
            f.write(f"\n")

        f.write(f"### Resumo de Exclusões por Pasta/Tipo:\n")
        exclusion_counts = {}
        for x in ignored_files:
            reason = x['reason']
            # group reasons a bit
            key = reason
            if reason.startswith("Excluded directory:"):
                key = "Diretórios de ambiente/sistema (.git, .venv, etc.)"
            elif reason.startswith("Contains sensitive pattern"):
                key = "Padrão sensitivo (chaves, senhas, tokens)"
            elif reason.startswith("Dataset file exceeds"):
                key = "Amostra de dataset acima do limite"
            elif reason.startswith("File format or path did not match"):
                key = "Formato de arquivo ignorado"
            exclusion_counts[key] = exclusion_counts.get(key, 0) + 1
            
        f.write(f"| Motivo de Exclusão | Quantidade de Arquivos |\n")
        f.write(f"| --- | --- |\n")
        for key, count in exclusion_counts.items():
            f.write(f"| {key} | {count} |\n")
        f.write(f"\n")

        # Guidance on what to copy/paste to AI
        f.write(f"## Arquivos Recomendados para Enviar à outra IA\n\n")
        f.write(f"Para obter a melhor ajuda na otimização de parâmetros e compreensão do pipeline, envie:\n")
        f.write(f"1. **`01_codigo_configuracoes_relatorios.zip`**: Contém todo o código-fonte principal, configurações YAML e relatórios de métricas.\n")
        f.write(f"2. **`04_logs_complementares.zip`**: Fornece o comportamento detalhado da execução atual.\n")
        f.write(f"3. **`RELATORIO_FINAL_PACOTE.md`** (este arquivo): Como sumário para a IA entender a estrutura do projeto.\n")
        
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Gerador de pacotes de auditoria para o projeto.")
    parser.add_argument('--project-root', type=str, default=None, help="Caminho raiz do projeto.")
    parser.add_argument('--output', type=str, default=None, help="Diretório de saída para os pacotes ZIP.")
    parser.add_argument('--latest', action='store_true', help="Seleciona a execução (run_id) mais recente automaticamente.")
    parser.add_argument('--run-id', type=str, default=None, help="Especifica um run_id para isolamento.")
    parser.add_argument('--max-zip-mb', type=int, default=450, help="Tamanho máximo de cada ZIP em MB.")
    parser.add_argument('--dry-run', action='store_true', help="Mostra o plano de compactação sem criar arquivos.")
    parser.add_argument('--verbose', action='store_true', help="Habilita mensagens detalhadas de log.")
    args = parser.parse_args()

    # Toggle logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 1. Resolve Project Root
    script_dir = Path(__file__).resolve().parent
    root = Path(args.project_root) if args.project_root else find_project_root(script_dir)
    logger.info(f"Projeto localizado na raiz: {root}")

    # 2. Load YAML Config
    config_path = root / "audit_package_config.yaml"
    config = load_config(config_path)
    if args.max_zip_mb != 450:
        config['max_zip_mb'] = args.max_zip_mb
    logger.debug(f"Configuração carregada: {config}")

    # 3. Resolve output folder (timestamp-based by default)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = root / f"pacote_auditoria_{timestamp}"

    # 4. Resolve Run ID
    run_candidates = scan_run_ids(root)
    if run_candidates:
        logger.info(f"Execuções (Run IDs) detectadas no projeto: {list(run_candidates.keys())}")
    else:
        logger.info("Nenhuma execução particionada (Run ID) encontrada no projeto. Usando layout plano.")
        
    try:
        selected_run_id = select_run_id(run_candidates, args.run_id, args.latest)
        if selected_run_id:
            logger.info(f"Execução selecionada para auditoria: {selected_run_id}")
    except ValueError as e:
        logger.error(f"Erro na validação da execução: {e}")
        sys.exit(1)

    # 5. Scan project files
    logger.info("Escaneando arquivos do projeto...")
    included_files, ignored_files = scan_files(root, config, selected_run_id, output_dir)
    logger.info(f"Escaneamento concluído: {len(included_files)} arquivos selecionados, {len(ignored_files)} ignorados.")

    if args.dry_run:
        logger.info("===== DRY RUN - NENHUM ZIP SERÁ CRIADO =====")

    # 6. Pack packages
    logger.info(f"Compactando arquivos em {output_dir}...")
    created_packages = pack_zips(root, included_files, ignored_files, selected_run_id, output_dir, config, args.dry_run)

    # 7. Integrity Tests
    integrity_failures = []
    if not args.dry_run:
        logger.info("Executando testes de integridade nos ZIPs criados...")
        integrity_failures = run_zip_integrity_checks(created_packages, root)
        if integrity_failures:
            logger.error("Falhas de integridade detectadas:")
            for fail in integrity_failures:
                logger.error(f" - {fail}")
        else:
            logger.info("Todos os testes de integridade passaram com sucesso!")

    # 8. Write final report
    report_file = write_final_report(root, created_packages, ignored_files, selected_run_id, output_dir, integrity_failures)
    logger.info(f"Relatório de auditoria gerado em: {report_file}")

    if integrity_failures:
        logger.error("A geração do pacote de auditoria falhou nos testes de integridade. Corrija os erros listados no relatório.")
        sys.exit(1)
        
    logger.info("Sucesso! Pacotes de auditoria gerados.")

if __name__ == '__main__':
    main()
