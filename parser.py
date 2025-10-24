import pandas as pd
import orjson as json
import os
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 로깅 설정 (print 대신 사용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(input_filename, output_dir):
    """
    CSV 파일 하나를 읽고, 'd' 컬럼의 JSON을 파싱하여 펼친 후,
    지정된 output 폴더에 저장합니다.
    (개선) 'itms'와 같은 중첩 리스트를 추가로 펼칩니다 (행으로 확장).
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
                    logging.warning(f"Failed to parse JSON string in {base_filename}: {x[:50]}...")
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
        flattened_json_df = pd.json_normalize(json_series)
        flattened_json_df.index = df.index # 원본 인덱스 유지

        # 5. 원본 데이터프레임과 결합 후, 불필요한 'd' 컬럼 제거
        df_expanded = pd.concat([df.drop(columns=['d']), flattened_json_df], axis=1)

        # 6. (신규) 중첩된 list 컬럼(itms)을 추가로 펼치는 로직
        list_columns_to_expand = ['pdu.prm.itms', 'pdu.dat.itms']
        
        for col_name in list_columns_to_expand:
            if col_name in df_expanded.columns:
                logging.info(f"Expanding nested list in '{col_name}' for {base_filename}...")
                
                # 1. 비어있는 리스트([]) / 비어있는 dict({}) / None / NaN 등을 pd.NA로 변환
                #    explode가 올바르게 작동하도록 함
                def clean_list(x):
                    if isinstance(x, list) and len(x) > 0:
                        return x
                    return pd.NA
                    
                df_expanded[col_name] = df_expanded[col_name].apply(clean_list)
                
                # 이 컬럼에 유효한 데이터가 하나도 없으면(전부 NA) 스킵
                if df_expanded[col_name].isna().all():
                    logging.info(f"Skipping empty list column '{col_name}'.")
                    continue

                # 2. 'explode'를 사용하여 리스트의 각 항목을 별도 행으로 분리
                #    (인덱스가 복제됨. 예: 1, 1, 2, 3, 3, 3)
                df_expanded = df_expanded.explode(col_name)
                
                # 3. 이제 'col_name' 컬럼에는 dict 또는 pd.NA가 들어있음.
                #    이 dict를 다시 normalize하여 펼침.
                normalized_col = pd.json_normalize(df_expanded[col_name])
                
                # 4. 펼쳐진 컬럼들에 접두사(prefix)를 붙여 이름 충돌 방지
                #    (예: 'pdu.prm.itms.syn', 'pdu.dat.itms.rc')
                normalized_col = normalized_col.add_prefix(f"{col_name}.")
                
                # 5. 원본 'col_name' 컬럼(이제 dict가 든)을 제거하고,
                #    펼쳐진 새 컬럼들을 *인덱스* 기준으로 join.
                #    (join은 인덱스 기준이 기본값)
                df_expanded = df_expanded.drop(columns=[col_name]).join(normalized_col)

        # 7. (수정) 결과 파일 저장
        output_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_parsed.csv")
        # NaN 값을 빈 문자열('')로 저장 (파싱 결과의 빈 필드 표현)
        df_expanded.to_csv(output_filename, index=False, na_rep='')
        
        return f"Successfully processed '{base_filename}' -> '{os.path.basename(output_filename)}'"

    except Exception as e:
        # traceback을 포함하여 더 자세한 에러 로깅
        logging.error(f"Critical error processing '{os.path.basename(input_filename)}': {e}", exc_info=True)
        return f"Error processing '{os.path.basename(input_filename)}': {e}"

def main():
    # ====================================================================
    # argparse를 사용하여 커맨드라인에서 경로를 입력받습니다.
    # ====================================================================
    parser = argparse.ArgumentParser(description="Parse JSON data within CSV files in parallel.")
    parser.add_argument('--input-dir', type=str, default='./pcap/data', help='Directory containing input CSV files.')
    parser.add_argument('--output-dir', type=str, default='./pcap/output', help='Directory to save processed CSV files.')
    args = parser.parse_args()

    # output 폴더가 없으면 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 처리할 파일 목록 가져오기
    files_to_process = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.endswith('.csv') and '_parsed' not in f
    ]

    if not files_to_process:
        logging.warning(f"No .csv files to process in '{args.input_dir}'.")
        return

    # 시스템의 CPU 코어 수를 워커 수로 사용
    num_workers = multiprocessing.cpu_count()
    logging.info(f"Starting ProcessPoolExecutor with {num_workers} workers for {len(files_to_process)} files.")
    
    # ProcessPoolExecutor를 사용하여 여러 파일을 병렬로 처리
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 각 파일 처리 함수에 output_dir 인자를 함께 전달
        futures = [executor.submit(process_file, filename, args.output_dir) for filename in files_to_process]
        
        for future in futures:
            result = future.result()
            if result:  # 에러가 발생했거나 처리할 데이터가 없는 경우는 None을 반환하므로 체크
                logging.info(result)
    
    logging.info("All processing finished.")

if __name__ == "__main__":
    main()
