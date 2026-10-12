"""Data flattener for nested Confirmit survey response records.

Converts nested loops (LoopT1, LpPSPrices) and complex question encodings
(compound grids, multi-choice true/false maps) into flat, queryable dictionaries.
"""
from typing import Any, Dict, List

def flatten_respondent(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single Confirmit respondent record into a flat key-value dict.
    
    Args:
        record: Raw respondent dictionary containing Confirmit nested structures
                (root variables, compound grids, multi-choice sets, and loop arrays).
                
    Returns:
        A flattened dictionary where:
        - Root scalars remain directly accessible by their original variable names.
        - Loop blocks are expanded with prefixed keys: '{loop_name}_{iteration}_{var_name}'.
        - Multi-choice choices are mapped to boolean flags '{var}_{code}' and a list of selected codes.
        - Grids are unpacked into sub-item keys '{grid}_{item}' and numeric sums '{grid}__sum'.
    """
    flat: Dict[str, Any] = {}

    for key, val in record.items():
        # Handle Confirmit loop blocks (e.g. LoopT1 and LpPSPrices)
        if key in ("LoopT1", "LpPSPrices"):
            if isinstance(val, list):
                flat[f"{key}__count"] = len(val)
                # Unpack each iteration in the loop array
                for idx, loop_item in enumerate(val, start=1):
                    # Use iteration key code if present, otherwise fallback to numeric index
                    loop_key = loop_item.get(key, str(idx))
                    for lk, lv in loop_item.items():
                        if lk == key:
                            continue
                        # Create unique prefixed key for each loop variable
                        prefix = f"{key}_{loop_key}_{lk}"
                        _flatten_value(prefix, lv, flat)
        else:
            # Flatten standard root variables
            _flatten_value(key, val, flat)

    return flat


def _flatten_value(prefix: str, val: Any, flat: Dict[str, Any]) -> None:
    """Helper to flatten individual values depending on their data structure.
    
    Args:
        prefix: Current field name or hierarchically prefixed key name.
        val: Value to process (None, scalar, dict, or list).
        flat: Target dictionary accumulating flattened key-value pairs.
    """
    # 1. Null / None values
    if val is None:
        flat[prefix] = None

    # 2. Primitive scalar types (int, float, string, boolean)
    elif isinstance(val, (int, float, str, bool)):
        flat[prefix] = val

    # 3. Dictionaries: Could be Confirmit multi-choice or compound numeric/grid
    elif isinstance(val, dict):
        # Case 3A: Confirmit Multi-choice representation {"true": ["1", "2"], "false": ["3"]}
        if "true" in val or "false" in val:
            true_codes = [str(c) for c in val.get("true", [])] if isinstance(val.get("true"), list) else []
            # Store list of selected option codes for fast set comparisons
            flat[prefix] = true_codes
            flat[f"{prefix}__selected_count"] = len(true_codes)
            
            # Unpack selected options as True flags
            for code in true_codes:
                flat[f"{prefix}_{code}"] = True
                
            # Unpack unselected options as False flags
            false_codes = [str(c) for c in val.get("false", [])] if isinstance(val.get("false"), list) else []
            for code in false_codes:
                flat[f"{prefix}_{code}"] = False
                
            # Store open-ended 'Other' text if present
            if "96$other" in val:
                flat[f"{prefix}_96_other"] = val["96$other"]

        # Case 3B: Compound numeric questions or grid answers {"1": 10, "2": 20, ...}
        else:
            # Preserve the dictionary reference for question-level access
            flat[prefix] = val
            numeric_vals = []
            
            for sub_k, sub_v in val.items():
                # Handle 'Other specify' text field
                if sub_k.endswith("$other"):
                    flat[f"{prefix}_{sub_k}"] = sub_v
                    continue
                    
                sub_key = f"{prefix}_{sub_k}"
                if isinstance(sub_v, (int, float)):
                    flat[sub_key] = sub_v
                    numeric_vals.append(sub_v)
                elif isinstance(sub_v, str):
                    # Attempt numeric cast for string-encoded numbers
                    try:
                        num = float(sub_v)
                        flat[sub_key] = num
                        numeric_vals.append(num)
                    except ValueError:
                        flat[sub_key] = sub_v
                else:
                    flat[sub_key] = sub_v

            # Calculate allocation totals and item counts for validation checks
            if numeric_vals:
                flat[f"{prefix}__sum"] = sum(numeric_vals)
                flat[f"{prefix}__count"] = len(numeric_vals)
                # For single-part numeric compounds (e.g. S5: {"1": 42.0}), provide direct scalar alias
                if len(val) == 1 and "1" in val and len(numeric_vals) == 1:
                    flat[f"{prefix}__val"] = numeric_vals[0]

    # 4. Lists (e.g., arrays of strings or codes)
    elif isinstance(val, list):
        flat[prefix] = val
        flat[f"{prefix}__count"] = len(val)

    # 5. Fallback for any other type
    else:
        flat[prefix] = val


def flatten_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten an entire collection of respondent records.
    
    Args:
        records: List of raw respondent records from responses_n*.json.
        
    Returns:
        List of flattened dictionaries ready for QC rule evaluation and agent queries.
    """
    return [flatten_respondent(r) for r in records]
