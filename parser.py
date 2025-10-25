import pandas as pd
import orjson as json
import os
import argparse
import logging
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional, Tuple, Set

# ====================================================================
# 로깅 설정
# ====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================================================================
# 전역 변수 (병렬 작업자 초기화를 위해 사용)
# ====================================================================
global_asset_ip_map: Dict[str, str] = {}
# (신규) 유선 레지스터 맵
global_wired_register_map: Dict[Tuple[str, str], str] = {}

# ====================================================================
# 자산 매핑 관련 헬퍼 함수
# (asset_mapper.py 로직 통합)
# ====================================================================

# 자산IP 시트의 장치 이름과 다른 시트의 컬럼 이름을 연결하는 맵
# {시트 컬럼명: 자산IP 시트의 공식 자산명}
DEVICE_NAME_MAP = {
    # 유선 (Output 시트 기준)
    'LS 산전 주소': 'LS Electric PLC',
    'SIEMENS 주소': 'Simens PLC', # '자산IP.csv'에 'Simens'로 되어 있음
    '미쯔비시 PLC 주소': 'Mitsubishi PLC',
    '히든 미쯔비시 PLC 주소': 'hidden PLC',
    
    # (신규) 유선 (Input 시트 변형)
    ' LS PLC 주소': 'LS Electric PLC', # 띄어쓰기 시작
    '미쯔비시 PLC주소': 'Mitsubishi PLC', # 띄어쓰기 없음

    '탈부착 월 PLC 주소(LS XGI)': '탈부착 월 PLC (LS XGI)', # IP 목록에 없음
    '탈부착 월 PLC 주소(미쓰비시)': '탈부착 월 PLC (미쓰비시)', # IP 목록에 없음

    # 무선 (현재 레지스터 매핑에서 사용 안 함)
    'LSE': 'LS PLC', 
    'Siemens': 'SIEMENS PLC'
}

def clean_ip(ip_str: str) -> Optional[str]:
    """
    "192,168.10.25", "modbus: 192.168.1.22/502", "192.1681.12" 같은
    다양한 IP 문자열을 정리합니다.
    """
    if not isinstance(ip_str, str):
        return None
    
    # "192,168.10.25" -> 192.168.10.25
    ip_str = ip_str.replace('"', '').replace(',', '.')
    
    # "modbus: 192.168.1.22/502" -> 192.168.1.22
    match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', ip_str)
    if match:
        ip_str = match.group(1)
    
    # "192.1681.12" (오타 수정)
    if ip_str == '192.1681.12':
        return '192.168.1.12'
        
    # 유효한 IP 형식인지 간단히 확인
    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip_str):
        return ip_str
        
    return None

def parse_asset_ip_sheet(excel_file_path: str, sheet_name: str = '자산IP') -> Dict[str, str]:
    """
    '자산IP' 시트(Excel)를 읽어 {IP: 자산명} 딕셔너리를 반환합니다.
    (키: IP, 값: 자산명으로 변경 - IP로 장치명을 찾아야 하므로)
    """
    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=2) # 3번째 줄을 헤더로 사용
    except Exception as e:
        logging.error(f"Error reading {excel_file_path} (Sheet: {sheet_name}): {e}", exc_info=True)
        return {}

    asset_ip_map: Dict[str, str] = {}
    
    # 1. 무선 자산 파트 (B, C 컬럼)
    wireless_assets = df[['구분', 'IP 정보']].copy()
    wireless_assets.columns = ['Asset_Name', 'IP_Address']
    
    # 2. 유선 자산 파트 (F, G 컬럼)
    wired_assets = df[['Device Name', 'IP']].copy()
    wired_assets.columns = ['Asset_Name', 'IP_Address']
    
    # 3. 데이터 합치기 및 정리
    all_assets = pd.concat([wireless_assets, wired_assets], ignore_index=True)
    all_assets.dropna(subset=['Asset_Name', 'IP_Address'], how='any', inplace=True)
    
    # 헤더 이름이 데이터로 포함된 경우 제거
    all_assets = all_assets[~all_assets['Asset_Name'].isin(['구분', 'Device Name'])]
    
    for _, row in all_assets.iterrows():
        asset_name = str(row['Asset_Name']).replace('\n', ' ').strip()
        ip_address = clean_ip(row['IP_Address'])
        
        if asset_name and ip_address:
            # 키: IP 주소, 값: 자산 이름
            asset_ip_map[ip_address] = asset_name
            
    return asset_ip_map

def load_wired_register_maps(excel_file_path: str) -> Dict[Tuple[str, str], str]:
    """
    '유선_Input/Output' 시트를 읽어 레지스터 맵을 생성합니다.
    (수정) Map Key: (Asset_Name, 번호_str)
    Map Value: Description
    """
    register_map: Dict[Tuple[str, str], str] = {}
    sheets_to_process = {
        '유선_Input': 'Input',
        '유선_Output': 'Output',
    }

    # --- 1. 유선 시트 처리 ---
    for sheet_name, direction in sheets_to_process.items():
        try:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            
            # '번호', '내용'을 제외한 컬럼을 장치 컬럼으로 간주
            device_cols = [col for col in df.columns if col not in ['번호', '내용']]
            
            for _, row in df.iterrows():
                description = row['내용']
                # (수정) '번호'를 정수형으로 읽고 문자열로 변환 (엑셀 형식 일관성)
                beonho_str = str(int(row['번호'])) 
                
                if pd.isna(description) or pd.isna(beonho_str):
                    continue
                
                for device_col_name in device_cols:
                    # 'LS 산전 주소' -> 'LS Electric PLC'
                    asset_name = DEVICE_NAME_MAP.get(device_col_name)
                    if not asset_name:
                        continue # 매핑 대상이 아닌 컬럼 (예: '탈부착...')
                    
                    # (수정) 이 컬럼에 값이 있는지(NA) 여부만 확인
                    if pd.isna(row[device_col_name]):
                        continue
                    
                    # (키): (자산명, 번호_문자열)
                    # (값): 설명
                    # 예: ('Mitsubishi PLC', '95') -> '스칼라 배출 카운터'
                    register_key = (asset_name, beonho_str)
                    register_map[register_key] = str(description)

        except Exception as e:
            logging.warning(f"Warning: Could not process sheet '{sheet_name}' in {excel_file_path}. {e}")

    # (수정) '무선' 시트 처리 로직 제거
    
    return register_map

# ====================================================================
# 병렬 작업자 초기화 함수
# ====================================================================

def init_worker(ip_map: Dict[str, str], wired_reg_map: Dict[Tuple[str, str], str]):
    """
    ProcessPoolExecutor의 각 작업자(worker)를 초기화합니다.
    """
    global global_asset_ip_map, global_wired_register_map
    global_asset_ip_map = ip_map
    global_wired_register_map = wired_reg_map # (수정)
    logging.info(f"Worker (PID: {os.getpid()}) initialized with {len(ip_map)} IP mappings and {len(wired_reg_map)} register mappings.")

# ====================================================================
# 핵심 파싱 로직 (단일 파일 처리)
# ====================================================================

def find_wired_description(row: pd.Series) -> str:
    """
    (수정) pdu.addr (요청) 또는 'pdu.regs.*' (응답)을 기반으로
    global_wired_register_map에서 레지스터 설명을 찾습니다.
    (신규 규칙) pdu.addr은 0-based offset, 엑셀 '번호'는 1-based.
               '번호' = pdu.addr + 1
    """
    
    # 1. Request 패킷 확인 (pdu.addr이 숫자일 경우)
    addr = row.get('pdu.addr')
    if pd.notna(addr):
        try:
            # Request는 PLC로 보낸 것 (dst_device가 PLC)
            asset_name = row.get('dst_device', 'Unknown')
            
            # (수정) '번호' = pdu.addr + 1
            beonho_str = str(int(addr) - 1) 
            
            key = (asset_name, beonho_str)
            desc = global_wired_register_map.get(key)
            if desc:
                # 예: pdu.addr: 2 -> 번호: 3 ('송신DATA교반기 인버터 RST')
                return desc 
        except Exception:
            pass # addr이 숫자가 아닐 경우 무시

    # 2. Response 패킷 확인 (dir == 'response' 이고 'pdu.regs.' 컬럼 확인)
    if row.get('dir') == 'response':
        # Response는 PLC가 보낸 것 (src_device가 PLC)
        asset_name = row.get('src_device', 'Unknown')
        descriptions = []
        
        # row.index (컬럼 이름들)을 순회
        for col_name in row.index:
            # 'pdu.regs.2', 'pdu.regs.95' 같은 컬럼을 찾음
            if col_name.startswith('pdu.regs.'):
                value = row[col_name]
                if pd.isna(value): # 값이 NaN이면 스킵
                    continue
                    
                # 'pdu.regs.2' -> '2'
                beonho_from_key_str = col_name.replace('pdu.regs.', '')
                
                try:
                    beonho_str = str(int(beonho_from_key_str) - 1) 
                except ValueError:
                    continue # 키가 숫자가 아니면 스킵

                key = (asset_name, beonho_str)
                desc = global_wired_register_map.get(key)
                
                if desc:
                    descriptions.append(f"{desc} (val: {value})")
        
        if descriptions:
            return "; ".join(descriptions)

    # 3. 매핑되는 것이 없을 경우
    return 'N/A'



def process_file(input_filename: str, output_dir: str) -> Optional[str]:
    """
    CSV 파일 하나를 읽고, 'd' 컬럼의 JSON을 파싱하여 펼친 후,
    IP와 레지스터를 매핑하여 저장합니다.
    """
    try:
        base_filename = os.path.basename(input_filename)
        logging.info(f"Processing '{base_filename}'...")

        # 1. CSV 파일 읽기
        df = pd.read_csv(input_filename)

        # 2. 'd' 컬럼이 비어있거나 유효한 JSON이 아닌 행 제거
        df.dropna(subset=['d'], inplace=True)
        if df.empty:
            logging.warning(f"'{base_filename}' has no data to process after dropping NaNs.")
            return None

        # 3. orjson으로 JSON 로드
        def safe_json_load(x):
            if not isinstance(x, str):
                return None
            try:
                # pandas가 CSV를 읽을 때 "..." -> {...} 로 올바르게 파싱한 경우
                return json.loads(x)
            except json.JSONDecodeError:
                try:
                    # pandas가 CSV를 잘못 읽어 '"{""prid...""}"' 처럼 문자열이 된 경우
                    return json.loads(x.strip('"').replace('""', '"'))
                except Exception:
                    # logging.warning(f"Failed to parse JSON string in {base_filename}: {x[:50]}...")
                    return None # 여전히 실패하면 None 반환

        json_series = df['d'].apply(safe_json_load)
        
        # 파싱에 실패한 행(None) 제거
        valid_json_indices = json_series.dropna().index
        if len(valid_json_indices) == 0:
            logging.warning(f"'{base_filename}' contains no valid JSON in 'd' column after parsing attempts.")
            return None
        
        # 파싱에 성공한 데이터만 유지 (SettingWithCopyWarning 방지)
        df = df.loc[valid_json_indices].copy()
        json_series = json_series.loc[valid_json_indices]

        # 4. pd.json_normalize로 1차 펼치기
        # (pdu.regs가 dict라면 { '2': 0, '3': 0 } -> 'pdu.regs.2': 0, 'pdu.regs.3': 0 으로 펼쳐짐)
        flattened_json_df = pd.json_normalize(json_series)
        flattened_json_df.index = df.index # 원본 인덱스 유지

        # 5. (신규) 중복 컬럼 제거 (JSON 내부의 'sip' 등이 원본 'sip'와 충돌하는 것 방지)
        original_cols = set(df.columns)
        json_cols = set(flattened_json_df.columns)
        duplicate_cols = original_cols.intersection(json_cols)
        
        if duplicate_cols:
            # 'd' 컬럼은 어차피 df에서 삭제될 것이므로 중복 목록에서 제외
            duplicate_cols.discard('d')
            if duplicate_cols:
                logging.info(f"Dropping duplicate columns from normalized JSON in '{base_filename}': {duplicate_cols}")
                flattened_json_df.drop(columns=list(duplicate_cols), inplace=True)

        # 6. 원본 데이터프레임과 결합 후, 불필요한 'd' 컬럼 제거
        df_expanded = pd.concat([df.drop(columns=['d']), flattened_json_df], axis=1)

        # 7. 중첩된 list 컬럼(itms)을 추가로 펼치는 로직
        # (수정) pdu.regs는 펼치지 않음 (딕셔너리 자체로 사용)
        list_columns_to_expand = ['pdu.prm.itms', 'pdu.dat.itms']
        
        for col_name in list_columns_to_expand:
            if col_name in df_expanded.columns:
                # logging.info(f"Expanding nested list/dict in '{col_name}' for {base_filename}...")
                
                def clean_list(x):
                    # pdu.regs는 dict, pdu.prm.itms는 list
                    if (isinstance(x, list) and len(x) > 0) or isinstance(x, dict):
                        return x
                    return pd.NA
                    
                df_expanded[col_name] = df_expanded[col_name].apply(clean_list)
                
                if df_expanded[col_name].isna().all():
                    # logging.info(f"Skipping empty list column '{col_name}'.")
                    continue

                # 'pdu.regs'는 dict이므로 explode하면 에러 발생.
                # list인 경우에만 explode 수행
                if isinstance(df_expanded[col_name].dropna().iloc[0], list):
                    df_expanded = df_expanded.explode(col_name)
                
                # dict 또는 list 안의 dict를 normalize
                try:
                    normalized_col = pd.json_normalize(df_expanded[col_name])
                    
                    # pdu.regs의 경우 컬럼 이름이 '2', '3', '4'...
                    # pdu.dat.itms의 경우 'rc', 'syn'...
                    # add_prefix가 유용함.
                    normalized_col = normalized_col.add_prefix(f"{col_name}.")
                    
                    df_expanded = df_expanded.drop(columns=[col_name]).join(normalized_col)
                except Exception as e:
                    logging.warning(f"Could not normalize column '{col_name}': {e}")


        # ====================================================================
        # 8. (신규) 자산 IP 매핑
        # ====================================================================
        # (IP 매핑을 레지스터 매핑보다 먼저 수행해야 src/dst_device를 사용 가능)
        if 'sip' in df_expanded.columns:
            df_expanded['src_device'] = df_expanded['sip'].map(global_asset_ip_map).fillna('Unknown')
        else:
            df_expanded['src_device'] = 'Unknown' # 컬럼이 없는 경우
            
        if 'dip' in df_expanded.columns:
            df_expanded['dst_device'] = df_expanded['dip'].map(global_asset_ip_map).fillna('Unknown')
        else:
            df_expanded['dst_device'] = 'Unknown' # 컬럼이 없는 경우

        # ====================================================================
        # 9. (수정) 레지스터 설명 매핑
        # ====================================================================
        # (pdu.regs는 펼쳐져서 'pdu.regs.2' 등으로 존재)
        df_expanded['asset_description'] = df_expanded.apply(find_wired_description, axis=1)

        # 10. 결과 파일 저장
        output_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_parsed.csv")
        # NaN 값을 빈 문자열('')로 저장
        df_expanded.to_csv(output_filename, index=False, na_rep='')
        
        return f"Successfully processed '{base_filename}' -> '{os.path.basename(output_filename)}'"

    except Exception as e:
        # traceback을 포함하여 더 자세한 에러 로깅
        logging.error(f"Critical error processing '{os.path.basename(input_filename)}': {e}", exc_info=True)
        return f"Error processing '{os.path.basename(input_filename)}': {e}"

# ====================================================================
# 메인 실행 함수
# ====================================================================

def main():
    # ====================================================================
    # argparse를 사용하여 커맨드라인에서 경로를 입력받습니다.
    # ====================================================================
    parser = argparse.ArgumentParser(description="Parse JSON data within CSV files, map Asset IPs, and map Registers.")
    
    parser.add_argument('--input-dir', type=str, default='../../TCP_Datagram_parser/output', 
                        help='Directory containing input CSV files.')
                        
    parser.add_argument('--output-dir', type=str, default='./pcap/output', 
                        help='Directory to save processed CSV files.')
                        
    parser.add_argument('--mapping-file', type=str, 
                        default='./2. 자산 목록 및 레지스터 매핑표.xlsx', 
                        help="Path to the Excel file containing '자산IP', '유선_Input/Output', '무선' sheets.")
    args = parser.parse_args()

    # --- 1. 필수 파일 확인 ---
    if not os.path.exists(args.mapping_file):
        logging.error(f"[오류] 필수 매핑 파일 '{args.mapping_file}'을(를) 찾을 수 없습니다.")
        logging.error("스크립트를 종료합니다.")
        return

    # --- 2. 매핑 정보 로드 (메인 프로세스에서 한 번만) ---
    logging.info(f"Loading IP mappings from '{args.mapping_file}' (자산IP 시트)...")
    # {IP: AssetName}
    asset_ip_map = parse_asset_ip_sheet(args.mapping_file)
    if not asset_ip_map:
        logging.error("자산 IP 정보를 로드하는 데 실패했습니다. 스크립트를 종료합니다.")
        return
    logging.info(f"-> {len(asset_ip_map)}개의 IP-자산 매핑을 로드했습니다.")

    logging.info(f"Loading Wired Register mappings from '{args.mapping_file}' (유선 시트)...")
    # (수정)
    # {(AssetName, 번호_Str): Description}
    wired_register_map = load_wired_register_maps(args.mapping_file)
    if not wired_register_map:
        logging.warning("유선 레지스터 매핑 정보를 로드하지 못했거나, 매핑이 비어있습니다.")
    logging.info(f"-> {len(wired_register_map)}개의 유선 레지스터-설명 매핑을 로드했습니다.")


    # --- 3. 처리할 파일 목록 가져오기 ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        files_to_process = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if f.endswith('.csv') and '_parsed' not in f
        ]
    except FileNotFoundError:
        logging.error(f"[오류] 입력 디렉터리 '{args.input_dir}'을(를) 찾을 수 없습니다.")
        logging.error("스크립트를 종료합니다.")
        return

    if not files_to_process:
        # (수정) 오타 수정
        logging.warning(f"No .csv files to process in '{args.input_dir}'.")
        return

    # --- 4. 병렬 처리 실행 ---
    num_workers = multiprocessing.cpu_count()
    logging.info(f"Starting ProcessPoolExecutor with {num_workers} workers for {len(files_to_process)} files.")
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        # init_worker 함수를 호출하여 각 작업자에 맵 전달
        initializer=init_worker,
        initargs=(asset_ip_map, wired_register_map) # (수정)
    ) as executor:
        
        futures = [executor.submit(process_file, filename, args.output_dir) for filename in files_to_process]
        
        for future in futures: 
            result = future.result()
            if result:
                logging.info(result)
    
    logging.info("All processing finished.")

if __name__ == "__main__":
    # Windows에서 ProcessPoolExecutor를 사용할 경우 필요
    multiprocessing.freeze_support() 
    
    # openpyxl 라이브러리 확인
    try:
        import openpyxl
    except ImportError:
        logging.error("="*60)
        logging.error(" [오류] 'openpyxl' 라이브러리가 설치되지 않았습니다.")
        logging.error(" Excel 파일을 읽으려면 이 라이브러리가 필요합니다.")
        logging.error(" 터미널에서 `pip install openpyxl` 명령어를 실행해주세요.")
        logging.error("="*60)
        exit(1)
        
    main()

