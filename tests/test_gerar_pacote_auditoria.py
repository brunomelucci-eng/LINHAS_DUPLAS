import os
import zipfile
import pytest
import tempfile
import csv
from pathlib import Path
import yaml

# Import functions to test
gpa = pytest.importorskip("gerar_pacote_auditoria")

def test_find_project_root(tmp_path):
    # Create mock project root with marker
    (tmp_path / "subdir" / "subsubdir").mkdir(parents=True)
    (tmp_path / "requirements.txt").touch()
    
    root = gpa.find_project_root(tmp_path / "subdir" / "subsubdir")
    assert root == tmp_path
    
    # If no marker, should fallback
    root2 = gpa.find_project_root(Path(tempfile.gettempdir()))
    assert root2.exists()

def test_load_config(tmp_path):
    cfg_file = tmp_path / "audit_package_config.yaml"
    cfg_data = {
        'max_zip_mb': 100,
        'sensitive_patterns': ['my_secret']
    }
    with open(cfg_file, 'w', encoding='utf-8') as f:
        yaml.dump(cfg_data, f)
        
    cfg = gpa.load_config(cfg_file)
    assert cfg['max_zip_mb'] == 100
    assert 'my_secret' in cfg['sensitive_patterns']
    assert 'venv' in cfg['exclude_directories']  # default merged

def test_calculate_sha256(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello World", encoding='utf-8')
    h = gpa.calculate_sha256(f)
    assert h == "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"

def test_truncate_log(tmp_path):
    f = tmp_path / "large.log"
    # Write 3000 lines
    lines = [f"Line {i}\n" for i in range(3000)]
    with open(f, 'w', encoding='utf-8') as file:
        file.writelines(lines)
        
    # Mock size limit to force truncation
    # max_log_mb = 0.0001 (very small, ~100 bytes)
    trunc_f = gpa.truncate_log(f, 0.0001)
    assert trunc_f != f
    assert trunc_f.exists()
    
    with open(trunc_f, 'r', encoding='utf-8') as file:
        trunc_lines = file.readlines()
    assert len(trunc_lines) == 2001
    assert "Line 0\n" in trunc_lines[0]
    assert "TRUNCATED" in trunc_lines[1000]
    assert "Line 2999\n" in trunc_lines[-1]
    
    trunc_f.unlink()

def test_scan_run_ids(tmp_path):
    (tmp_path / "outputs" / "run_abc").mkdir(parents=True)
    (tmp_path / "runs" / "run_123").mkdir(parents=True)
    (tmp_path / "outputs" / "dataset").mkdir(parents=True)  # standard folder, ignore
    
    runs = gpa.scan_run_ids(tmp_path)
    assert "run_abc" in runs
    assert "run_123" in runs
    assert "dataset" not in runs

def test_select_run_id():
    runs = {
        "run_abc": Path("outputs/run_abc"),
        "run_123": Path("runs/run_123")
    }
    # Specify ID
    assert gpa.select_run_id(runs, "run_abc", False) == "run_abc"
    with pytest.raises(ValueError):
        gpa.select_run_id(runs, "invalid", False)

def test_is_path_excluded(tmp_path):
    config = {
        'exclude_directories': ['.venv', '__pycache__'],
        'sensitive_patterns': ['.env', 'password']
    }
    # Excluded directory
    f1 = tmp_path / ".venv" / "script.py"
    assert gpa.is_path_excluded(f1, tmp_path, config, None)[0] is True
    
    # Sensitive file
    f2 = tmp_path / "password.txt"
    assert gpa.is_path_excluded(f2, tmp_path, config, None)[0] is True
    
    # Normal file
    f3 = tmp_path / "src" / "main.py"
    assert gpa.is_path_excluded(f3, tmp_path, config, None)[0] is False

def test_dataset_sampling_and_pairing(tmp_path):
    config = {
        'max_zip_mb': 100,
        'sample_limits': {'train': 2, 'val': 1, 'test': 1},
        'exclude_directories': [],
        'sensitive_patterns': [],
        'preferred_model_names': [],
        'include_extensions': ['.png', '.txt', '.npz']
    }
    # Create dummy images and labels
    (tmp_path / "dataset" / "images" / "train").mkdir(parents=True)
    (tmp_path / "dataset" / "labels" / "train").mkdir(parents=True)
    
    img1 = tmp_path / "dataset" / "images" / "train" / "im1.png"
    img2 = tmp_path / "dataset" / "images" / "train" / "im2.png"
    img3 = tmp_path / "dataset" / "images" / "train" / "im3.png"
    
    lbl1 = tmp_path / "dataset" / "labels" / "train" / "im1.txt"
    lbl2 = tmp_path / "dataset" / "labels" / "train" / "im2.txt"
    lbl3 = tmp_path / "dataset" / "labels" / "train" / "im3.txt"
    
    for f in [img1, img2, img3, lbl1, lbl2, lbl3]:
        f.write_text("dummy", encoding='utf-8')
        
    included, ignored = gpa.scan_files(tmp_path, config, None, None)
    
    selected_paths = [x['path'] for x in included if x['category'] == '02_amostras_dados']
    ignored_paths = [x['path'] for x in ignored if "sample limit" in x['reason']]
    
    assert img1 in selected_paths
    assert img2 in selected_paths
    assert lbl1 in selected_paths
    assert lbl2 in selected_paths
    assert img3 in ignored_paths
    assert lbl3 in ignored_paths

def test_dry_run_and_pack_zips(tmp_path):
    config = gpa.load_config(Path("non_existent.yaml"))
    config['max_zip_mb'] = 100
    config['sample_limits'] = {'train': 1, 'val': 1, 'test': 1}
    
    (tmp_path / "src").mkdir()
    code_f = tmp_path / "src" / "main.py"
    code_f.write_text("print('hello')", encoding='utf-8')
    
    output_dir = tmp_path / "audit_out"
    
    included, ignored = gpa.scan_files(tmp_path, config, None, output_dir)
    pkg_dry = gpa.pack_zips(tmp_path, included, ignored, None, output_dir, config, dry_run=True)
    
    assert len(pkg_dry) > 0
    assert "01_codigo_configuracoes_relatorios.zip" in pkg_dry
    assert not (output_dir / "01_codigo_configuracoes_relatorios.zip").exists()
    
    pkg_real = gpa.pack_zips(tmp_path, included, ignored, None, output_dir, config, dry_run=False)
    assert (output_dir / "01_codigo_configuracoes_relatorios.zip").exists()
    
    zip_path = output_dir / "01_codigo_configuracoes_relatorios.zip"
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        assert "MANIFESTO_AUDITORIA.md" in namelist
        assert "arquivos_incluidos.csv" in namelist
        assert "src/main.py" in namelist

def test_run_zip_integrity_checks(tmp_path):
    config = gpa.load_config(Path("non_existent.yaml"))
    
    (tmp_path / "src").mkdir()
    code_f = tmp_path / "src" / "main.py"
    code_f.write_text("print('hello')", encoding='utf-8')
    
    output_dir = tmp_path / "audit_out"
    included, ignored = gpa.scan_files(tmp_path, config, None, output_dir)
    
    created = gpa.pack_zips(tmp_path, included, ignored, None, output_dir, config, dry_run=False)
    
    failures = gpa.run_zip_integrity_checks(created, tmp_path)
    assert len(failures) == 0
