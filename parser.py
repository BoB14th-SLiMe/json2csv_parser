import pandas as pd
import orjson as json
import os
import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Tuple, Any, List # (수정) List 추가
import re # 정규식 모듈 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 전역 변수 (병렬 작업자 초기화용)
global_asset_ip_map: Dict[str, str] = {}
# (수정) 맵 구조 변경: { 자산명 -> {레지스터 주소 -> 설명} }
global_all_register_maps: Dict[str, Dict[str, str]] = {}


# --- 자산/레지스터 매핑 로직 (co.py 규칙 적용) ---

DEVICE_NAME_MAP = {
    # '자산IP' 시트 (유선)
    'LS Electric PLC': 'LS Electric PLC',
    'Simens PLC': 'Simens PLC',
    'Mitsubishi PLC': 'Mitsubishi PLC',
    'hidden PLC': 'hidden PLC',
    '탈부착 월 PLC 주소(LS XGI)': '탈부착 월 PLC 주소(LS XGI)',
    '탈부착 월 PLC 주소(미쓰비시)': '탈부착 월 PLC 주소(미쓰비시)',
    # (신규) '자산IP' 시트 (무선)
    '부착형 HMI': '부착형 HMI', # HMI도 자산일 수 있으므로 추가
    'WirelessHART Gateway': 'WirelessHART Gateway',
    'Wi-Fi AP': 'Wi-Fi AP',
    'Wi-Fi Client Module\n(LS PLC)': 'LS Electric PLC', # XGT IP 매핑 오류 수정
    'Wi-Fi Client Module\n(SIEMENS)': 'Simens PLC', # XGT IP 매핑 오류 수정
    '5G Router': '5G Router',
    'LS PLC': 'LS Electric PLC', # XGT IP 매핑 오류 수정
    # '유선_Input/Output' 시트
    '히든 미쯔비시 PLC 주소': 'hidden PLC',
    '미쯔비시 PLC 주소': 'Mitsubishi PLC',
    '미쯔비시 PLC주소': 'Mitsubishi PLC', 
    'LS 산전 주소': 'LS Electric PLC',
    ' LS PLC 주소': 'LS Electric PLC', 
    'SIEMENS 주소': 'Simens PLC',
    # '무선' 시트
    'LSE': 'LS Electric PLC', 
    'Siemens': 'Simens PLC', 
}

# (신규) co.py의 Modbus 오프셋 맵
# (규칙) key: (자산명, Function Code), value: (오프셋, 주소 타입)
MODBUS_OFFSET_MAP = {
    ('Mitsubishi PLC', 3): (400000, 'modbus'),
    ('Mitsubishi PLC', 4): (300000, 'modbus'),
    ('LS Electric PLC', 3): (400000, 'modbus'), # (가정)
    ('LS Electric PLC', 4): (300000, 'modbus'), # (가정)
    ('hidden PLC', 3): (400000, 'modbus'), # (가정)
    ('hidden PLC', 4): (300000, 'modbus'), # (가정)
    # S7 (Simens PLC)는 오프셋 대신 DB 번호를 사용
    ('Simens PLC', 3): (0, 's7'),
    ('Simens PLC', 4): (0, 's7'),
}

def clean_ip(ip_str: str) -> str:
    """ "192,168.10.25" 같은 비정상 IP를 "192.168.10.25"로 정리 """
    if isinstance(ip_str, str):
        return ip_str.replace(',', '.').strip().strip('"')
    return str(ip_str)

def parse_asset_ip_sheet(df_ip: pd.DataFrame) -> Dict[str, str]:
    """ '자산IP' 시트를 파싱하여 {IP -> 자산명} 딕셔너리 생성 """
    ip_map = {}
    
    # 1. <무선 ... IP 주소> 섹션
    df_wireless = df_ip.iloc[3:13, 1:3].dropna(how='all')
    df_wireless.columns = ['Asset_Name_Raw', 'IP']
    for _, row in df_wireless.iterrows():
        # (수정) \n (줄바꿈)이 있는 키를 처리하기 위해 .replace() 추가
        asset_name_raw_clean = str(row['Asset_Name_Raw']).replace('\r\n', '\n')
        asset_name = DEVICE_NAME_MAP.get(asset_name_raw_clean)
        if asset_name and pd.notna(row['IP']):
            ip = clean_ip(row['IP'])
            ip_map[ip] = asset_name

    # 2. <유선 ... IP 주소> 섹션
    df_wired = df_ip.iloc[3:13, 5:7].dropna(how='all')
    df_wired.columns = ['Asset_Name_Raw', 'IP']
    for _, row in df_wired.iterrows():
        asset_name_raw_clean = str(row['Asset_Name_Raw']).replace('\r\n', '\n') # (수정)
        asset_name = DEVICE_NAME_MAP.get(asset_name_raw_clean)
        if asset_name and pd.notna(row['IP']):
            ip = clean_ip(row['IP'])
            ip_map[ip] = asset_name
            
    # 3. 'hidden PLC'의 특수 IP 처리
    hidden_plc_ip = clean_ip(df_ip.iloc[10, 6]) # G11 (0-based 10, 6)
    if hidden_plc_ip:
         ip_map[hidden_plc_ip] = 'hidden PLC'
         
    logging.info(f"자산 IP 맵 생성 완료: {len(ip_map)}개 항목")
    return ip_map

def load_register_maps(excel_file: str) -> Dict[str, Dict[str, str]]:
    """
    (수정) '유선_Input', '유선_Output', '무선' 시트를 모두 읽어
    { 자산명 -> {레지스터 주소 -> 설명} } 맵을 생성
    """
    all_maps: Dict[str, Dict[str, str]] = {}
    sheet_names = ['유선_Input', '유선_Output', '무선']
    
    for sheet in sheet_names:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet, dtype=str)
            
            if '내용' not in df.columns:
                logging.warning(f"'{sheet}' 시트에 '내용' 컬럼이 없어 건너뜁니다.")
                continue
            
            df.dropna(subset=['내용'], inplace=True)
            
            asset_columns = [col for col in df.columns if col in DEVICE_NAME_MAP]
            
            for asset_col_name in asset_columns:
                asset_name = DEVICE_NAME_MAP[asset_col_name]
                
                if asset_name not in all_maps:
                    all_maps[asset_name] = {}
                    
                # '내용'과 '레지스터 주소' 컬럼만 추출
                register_map_df = df.dropna(subset=[asset_col_name])[['내용', asset_col_name]]
                
                for _, row in register_map_df.iterrows():
                    desc = row['내용'].strip()
                    # (수정) 레지스터 주소 자체를 키로 사용
                    addr_str = str(row[asset_col_name]).strip() 
                    
                    if addr_str and desc:
                        # (수정) Modbus 주소가 '400001.0'처럼 읽히는 것을 방지
                        if re.match(r'^\d+(\.0)?$', addr_str): # '400001' 또는 '400001.0'
                            addr_str = addr_str.split('.')[0]

                        # (수정) S7/XGT 주소 정리
                        addr_str = addr_str.replace('"', '')
                        # (수정) XGT 주소 '[PLC2]%...' -> '%...'
                        if addr_str.startswith('['):
                             addr_str = addr_str.split(']')[-1]
                        # (수정) XGT 주소 '%MW401 (D500-히든)' -> '%MW401'
                        addr_str = addr_str.split(' ')[0] 
                        
                        if addr_str in all_maps[asset_name]:
                             logging.warning(f"중복된 레지스터 맵 키: ({asset_name}, {addr_str}). (시트: {sheet})")
                        all_maps[asset_name][addr_str] = desc
                         
            logging.info(f"'{sheet}' 시트 로드 완료. (키: 레지스터 주소)")
            
        except Exception as e:
            logging.error(f"'{sheet}' 시트 로드 중 오류: {e}")
            
    total_mappings = sum(len(v) for v in all_maps.values())
    logging.info(f"전체 레지스터 맵 생성 완료: {len(all_maps)}개 자산, {total_mappings}개 매핑")
    return all_maps

def translate_xgt_byte_addr(byte_addr: str) -> Optional[str]:
    """
    (신규) LS XGT FEnet 바이트 주소 (예: %DB001000)를 워드 주소 (예: D500)로 변환
    규칙 1: %DB -> D, %MB -> M, %PB -> P
    규칙 2: 숫자 // 2
    """
    if not isinstance(byte_addr, str) or not byte_addr.startswith('%') or len(byte_addr) < 4:
        return None
    
    mem_type = byte_addr[1:3] # 'DB', 'MB', 'PB'
    num_str = byte_addr[3:]
    
    word_type = None
    if mem_type == 'DB':
        word_type = 'D'
    elif mem_type == 'MB':
        word_type = 'M'
    elif mem_type == 'PB':
        word_type = 'P'
    else:
        return None # 모르는 바이트 타입 (예: %MW, %IX)

    try:
        byte_num = int(num_str)
        word_num = byte_num // 2
        return f"{word_type}{word_num}"
    except ValueError:
        return None

def find_asset_description(row: pd.Series) -> str:
    """
    (수정) co.py의 변환 규칙 (S7, Modbus, XGT)을 적용하여 레지스터 설명을 찾는 함수
    """
    try:
        direction = row.get('dir')
        asset_name = 'Unknown'
        
        # 1. 자산명 결정
        if direction == 'request':
            asset_name = row.get('dst_device', 'Unknown')
        elif direction == 'response':
            asset_name = row.get('src_device', 'Unknown')
        else:
            return 'N/A' # dir 정보가 없으면 매핑 불가

        if asset_name == 'Unknown' or asset_name not in global_all_register_maps:
            return 'N/A' # 매핑 대상 자산이 아니거나 맵이 없음

        register_map = global_all_register_maps[asset_name]
        
        # --- 프로토콜별 주소 '번역' ---
        
        lookup_addr_list: List[Tuple[str, Any]] = [] # (번역된 주소, 값)
        
        # 2. Function Code 추출 (Modbus/S7 확인용)
        fc = row.get('pdu.fc')
        fc_int = None
        if pd.notna(fc):
            try: 
                fc_int = int(fc)
            except ValueError: 
                pass # fc가 숫자가 아님

        # 3. SIEMENS S7 규칙 (Simens PLC)
        if asset_name == 'Simens PLC':
            db_num = row.get('pdu.prm.itms.db') # 단일 항목
            addr = row.get('pdu.prm.itms.addr')
            
            if pd.isna(db_num):
                db_num = row.get('pdu.prm.itms.0.db') # 리스트의 0번째 항목
                addr = row.get('pdu.prm.itms.0.addr')

            if pd.notna(db_num) and pd.notna(addr):
                try:
                    # (규칙) "DB{db},{addr}"
                    lookup_addr = f"DB{int(db_num)},{int(addr)}"
                    lookup_addr_list.append((lookup_addr, None)) # S7 Request는 값 모름
                except Exception:
                    pass
            # S7 Response (규칙 불명확)
            # elif direction == 'response':
            #    return 'N/A (S7 Response)'

        # 4. Modbus 규칙 (Mitsubishi, LS, hidden)
        offset_key = (asset_name, fc_int)
        if offset_key in MODBUS_OFFSET_MAP:
            offset, addr_type = MODBUS_OFFSET_MAP[offset_key]
            
            if addr_type == 'modbus':
                # 4a. Modbus Request (pdu.addr)
                addr = row.get('pdu.addr')
                if pd.notna(addr):
                    try:
                        # (규칙) offset + pdu.addr + 1
                        lookup_addr = str(int(offset + addr + 1))
                        lookup_addr_list.append((lookup_addr, None)) # Request는 값 모름
                    except Exception:
                        pass

                # 4b. Modbus Response (pdu.regs.*)
                elif direction == 'response':
                    for col_name in row.index:
                        if col_name.startswith('pdu.regs.'):
                            value = row[col_name]
                            if pd.isna(value): continue
                            try:
                                reg_key_int = int(col_name.replace('pdu.regs.', ''))
                                # (규칙) offset + reg_key + 1
                                lookup_addr = str(int(offset + reg_key_int + 1))
                                lookup_addr_list.append((lookup_addr, value))
                            except Exception:
                                continue

        # 5. LS XGT FEnet 규칙 (LS Electric PLC, XGI)
        elif 'LS' in asset_name or 'XGI' in asset_name:
            xgt_addr_str = None
            xgt_val = None
            
            # 1순위: pdu.var.nm (바이트 주소 문자열 -> 번역 필요)
            addr_nm = row.get('pdu.var.nm')
            if pd.notna(addr_nm):
                xgt_addr_str = translate_xgt_byte_addr(str(addr_nm))
                if direction == 'response' and pd.notna(row.get('pdu.dat.val')):
                     xgt_val = row.get('pdu.dat.val') # (가정)
            
            # 2순위: pdu.var (이미 워드 주소 문자열)
            if not xgt_addr_str and pd.notna(row.get('pdu.var')):
                xgt_addr_str = str(row.get('pdu.var'))
                if direction == 'response' and pd.notna(row.get('pdu.dat.itms.data')):
                     xgt_val = row.get('pdu.dat.itms.data') # (기존 가정)
            
            # 3순위: pdu.prm.itms.var (블록)
            if not xgt_addr_str and pd.notna(row.get('pdu.prm.itms.var')):
                 xgt_addr_str = str(row.get('pdu.prm.itms.var'))
                 if direction == 'response' and pd.notna(row.get('pdu.dat.itms.0.data')):
                     xgt_val = row.get('pdu.dat.itms.0.data') # (기존 가정)

            # 4순위: pdu.prm.itms.0.var (리스트의 첫 변수)
            if not xgt_addr_str and pd.notna(row.get('pdu.prm.itms.0.var')):
                 xgt_addr_str = str(row.get('pdu.prm.itms.0.var'))
                 if direction == 'response' and pd.notna(row.get('pdu.dat.itms.0.data')):
                     xgt_val = row.get('pdu.dat.itms.0.data') # (기존 가정)

            if xgt_addr_str:
                lookup_addr = xgt_addr_str.split(' ')[0] # '%MW401 (D500..)' 정리
                lookup_addr_list.append((lookup_addr, xgt_val))

        # --- 매핑 수행 ---
        if not lookup_addr_list:
            return 'N/A'

        descriptions = []
        for lookup_addr, value in lookup_addr_list:
            desc = register_map.get(lookup_addr)
            if desc:
                if value is not None and pd.notna(value):
                    descriptions.append(f"{desc} (val: {value})")
                else:
                    descriptions.append(desc) # Request
            else:
                descriptions.append(f"N/A (Addr: {lookup_addr})")
        
        return "; ".join(descriptions)

    except Exception as e:
        # logging.warning(f"find_asset_description 오류: {e} - 행: {row.get('tid', 'N/A')}")
        return f'N/A (Error)'


# --- 병렬 처리 작업자 ---

def init_worker(ip_map: Dict[str, str], reg_maps: Dict[str, Dict[str, str]]):
    """ 병렬 작업자 프로세스 초기화 """
    global global_asset_ip_map
    global global_all_register_maps
    global_asset_ip_map = ip_map
    global_all_register_maps = reg_maps
    # logging.info(f"Worker (PID: {os.getpid()}) initialized with {len(ip_map)} IP mappings and {len(reg_maps)} register map keys.")

def process_file(input_filename: str, output_dir: str) -> Optional[str]:
    """
    CSV 파일 하나를 읽고, 'd' 컬럼의 JSON을 파싱하여 펼친 후,
    IP 매핑 및 레지스터 매핑을 수행하고 저장합니다.
    """
    try:
        base_filename = os.path.basename(input_filename)
        logging.info(f"Processing '{base_filename}'...")

        # 1. CSV 파일 읽기
        df = pd.read_csv(input_filename)

        # 2. 'd' 컬럼 비우기
        df.dropna(subset=['d'], inplace=True)
        if df.empty:
            logging.warning(f"'{base_filename}' has no data to process after dropping NaNs.")
            return None

        # 3. orjson으로 JSON 로드
        def safe_json_load(x: Any) -> Optional[Dict]:
            if not isinstance(x, str):
                return None
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                try:
                    return json.loads(x.strip('"').replace('""', '"'))
                except Exception:
                    # logging.warning(f"Failed to parse JSON string in {base_filename}: {x[:50]}...")
                    return None

        json_series = df['d'].apply(safe_json_load)
        
        valid_json_indices = json_series.dropna().index
        if len(valid_json_indices) == 0:
            logging.warning(f"'{base_filename}' contains no valid JSON in 'd' column after parsing attempts.")
            return None
        
        df = df.loc[valid_json_indices].copy()
        json_series = json_series.loc[valid_json_indices]

        # 4. pd.json_normalize로 1차 펼치기
        # max_level=None 유지 (S7, XGT의 중첩 구조를 펼쳐야 함)
        flattened_json_df = pd.json_normalize(json_series, max_level=None)
        flattened_json_df.index = df.index

        # 5. 중복 컬럼 제거
        original_cols = set(df.columns)
        json_cols_to_drop = [col for col in flattened_json_df.columns if col in original_cols]
        if json_cols_to_drop:
            flattened_json_df.drop(columns=json_cols_to_drop, inplace=True)

        # 6. 원본과 결합
        df_expanded = pd.concat([df.drop(columns=['d']), flattened_json_df], axis=1)

        # 7. IP 주소를 자산명으로 매핑
        if 'sip' in df_expanded.columns:
            df_expanded['src_device'] = df_expanded['sip'].map(global_asset_ip_map).fillna('Unknown')
        else:
            df_expanded['src_device'] = 'Unknown'
            
        if 'dip' in df_expanded.columns:
            df_expanded['dst_device'] = df_expanded['dip'].map(global_asset_ip_map).fillna('Unknown')
        else:
            df_expanded['dst_device'] = 'Unknown'

        # 8. (수정) 레지스터 주소/번호를 자산 설명으로 매핑
        if not global_all_register_maps:
            df_expanded['asset_description'] = 'N/A (Map Not Loaded)'
        else:
            # .apply()가 행(Series)을 함수로 전달
            df_expanded['asset_description'] = df_expanded.apply(find_asset_description, axis=1)
        
        # 9. (주석 처리) 중첩 리스트 펼치기 (pdu.prm.itms)
        # S7/XGT 매핑을 위해 pdu.prm.itms가 펼쳐져야 함 (json_normalize가 처리)
        # 만약 S7/XGT pdu.prm.itms가 리스트이고 모든 항목을 봐야 한다면,
        # find_asset_description 전에 explode 로직이 필요함.
        # list_columns_to_expand = ['pdu.prm.itms'] 
        
        # 10. 결과 파일 저장
        output_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_parsed.csv")
        df_expanded.to_csv(output_filename, index=False, na_rep='')
        
        return f"Successfully processed '{base_filename}' -> '{os.path.basename(output_filename)}'"

    except Exception as e:
        logging.error(f"Critical error processing '{os.path.basename(input_filename)}': {e}", exc_info=True)
        return f"Error processing '{os.path.basename(input_filename)}': {e}"

# --- 메인 실행 로직 ---

def main():
    parser = argparse.ArgumentParser(description="Parse JSON data within CSV files and map asset/register info.")
    
    parser.add_argument('--input-dir', type=str, default='../../TCP_Datagram_parser/output',
                        help='Directory containing input CSV files.')
    
    parser.add_argument('--mapping-file', type=str, default='2. 자산 목록 및 레지스터 매핑표.xlsx', 
                        help="Path to the Excel file containing '자산IP', '유선_Input', '유선_Output', '무선' sheets.")
                        
    parser.add_argument('--output-dir', type=str, default='./pcap/output', 
                        help="Directory to save processed CSV files.")
    
    args = parser.parse_args()

    # --- 0. 필수 파일 확인 ---
    if not os.path.exists(args.mapping_file):
        logging.error(f"[오류] 매핑 파일(Excel)을 찾을 수 없습니다: {args.mapping_file}")
        return
    if not os.path.isdir(args.input_dir):
        logging.error(f"[오류] 입력 디렉터리(CSV)를 찾을 수 없습니다: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. (메인 프로세스) Excel 매핑 파일 미리 로드 ---
    logging.info(f"Loading IP mappings from '{args.mapping_file}' (자산IP 시트)...")
    try:
        df_ip = pd.read_excel(args.mapping_file, sheet_name='자산IP', header=None)
        asset_ip_map = parse_asset_ip_sheet(df_ip)
    except Exception as e:
        if 'openpyxl' in str(e):
            logging.error("="*50)
            logging.error(" [오류] 'openpyxl' 라이브러리가 필요합니다. ")
            logging.error(" 터미널에서 'pip install openpyxl'을 실행해주세요. ")
            logging.error("="*50)
        else:
            logging.error(f"'자산IP' 시트 로드 실패: {e}", exc_info=True)
        return

    logging.info(f"Loading register mappings from '{args.mapping_file}' (유선/무선 시트)...")
    # (수정) '레지스터 주소'를 키로 사용하는 새 함수 호출
    all_register_maps = load_register_maps(args.mapping_file)

    if not asset_ip_map or not all_register_maps:
        logging.error("매핑 정보(IP 또는 레지스터)를 로드하지 못했습니다. 스크립트를 종료합니다.")
        return

    # --- 2. 처리할 CSV 파일 목록 가져오기 ---
    files_to_process = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.endswith('.csv') and '_parsed' not in f
    ]

    if not files_to_process:
        logging.warning(f"No .csv files to process in '{args.input_dir}'.")
        return

    # --- 3. 병렬 처리 실행 ---
    num_workers = multiprocessing.cpu_count()
    logging.info(f"Starting ProcessPoolExecutor with {num_workers} workers for {len(files_to_process)} files.")
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(asset_ip_map, all_register_maps) # 수정된 맵 전달
    ) as executor:
        
        futures = [executor.submit(process_file, filename, args.output_dir) for filename in files_to_process]
        
        for future in futures:
            try:
                result = future.result()
                if result:
                    logging.info(result)
            except Exception as e:
                logging.error(f"A worker process failed: {e}", exc_info=True)
    
    logging.info(f"All processing finished.")

if __name__ == "__main__":
    try:
        import openpyxl
    except ImportError:
        logging.warning("="*50)
        logging.warning(" [경고] 'openpyxl' 라이브러리가 없습니다. ")
        logging.warning(" Excel 매핑 파일을 읽기 위해 설치가 필요합니다. ")
        logging.warning(" > pip install openpyxl ")
        logging.warning("="*50)
        
    main()


