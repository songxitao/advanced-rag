import os
import json
import pytest
from tests.disguise_book_generator import run_disguise_pipeline

def test_run_disguise_pipeline():
    input_file = "E:/project/pyltp-books-master/pyltp-books-master/mybooks/Book/三国演义白话文"
    output_dir = "e:/project/advanced-rag/tests/temp_data/"
    
    # Paths of expected outputs
    alias_path = os.path.join(output_dir, "sanguo_aliases.json")
    disguised_path = os.path.join(output_dir, "三国演义白话文_disguised.txt")
    
    # Clean up output files if they exist to ensure clean state
    if os.path.exists(alias_path):
        os.remove(alias_path)
    if os.path.exists(disguised_path):
        os.remove(disguised_path)
        
    # Execute the disguise pipeline
    ret_alias, ret_disguised = run_disguise_pipeline(input_file, output_dir)
    
    # 1. Verify output paths and file existence
    assert ret_alias == alias_path
    assert ret_disguised == disguised_path
    assert os.path.exists(alias_path), "sanguo_aliases.json was not created"
    assert os.path.exists(disguised_path), "三国演义白话文_disguised.txt was not created"
    
    # 2. Verify JSON structure and contents
    with open(alias_path, 'r', encoding='utf-8') as f:
        aliases = json.load(f)
        
    assert isinstance(aliases, dict), "Aliases JSON should be a dictionary"
    assert len(aliases) > 0, "Aliases JSON should not be empty"
    
    # Verify core characters are represented in the keys or aliases (e.g. 刘备 or 曹操)
    core_character_found = any(key in aliases for key in ["刘备", "曹操", "关羽", "张飞", "诸葛亮"])
    assert core_character_found, "Core characters not found in aliases keys"
    
    # 3. Verify disguised text
    with open(disguised_path, 'r', encoding='utf-8') as f:
        disguised_text = f.read()
        
    assert len(disguised_text) > 0, "Disguised text should not be empty"
    
    # Verify disguise codes exist in the text
    assert "[角色_" in disguised_text, "Disguise codes [角色_x] not found in disguised text"
    
    # Collect all original names and aliases to verify they were fully replaced
    all_grouped_names = []
    for std_name, alias_list in aliases.items():
        all_grouped_names.append(std_name)
        if isinstance(alias_list, list):
            all_grouped_names.extend(alias_list)
        else:
            all_grouped_names.append(alias_list)
            
    # Filter unique and non-empty names
    all_grouped_names = sorted(list(set([name.strip() for name in all_grouped_names if name.strip()])))
    
    # Assert that none of the mapped names/aliases exist in the disguised text
    failed_names = []
    for name in all_grouped_names:
        if name in disguised_text:
            failed_names.append(name)
            
    assert not failed_names, f"The following names were not fully replaced: {failed_names}"
