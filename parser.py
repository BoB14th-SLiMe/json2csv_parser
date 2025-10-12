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

        # 3. (핵심 개선) orjson으로 JSON 로드 후, pd.json_normalize로 한 번에 펼치기
        #    - 기존 flatten_json + apply 조합보다 훨씬 빠르고 효율적입니다.
        #    - 문자열 클리닝 로직은 데이터 형태에 따라 유지합니다.
        json_series = df['d'].apply(lambda x: json.loads(x.strip('"').replace('""', '"')))
        
        # 인덱스를 유지하여 나중에 원본 데이터프레임과 정렬을 맞춥니다.
        flattened_json_df = pd.json_normalize(json_series)
        flattened_json_df.index = df.index

        # 4. 기존 데이터프레임과 결합 후, 불필요한 'd' 컬럼 제거
        df_expanded = pd.concat([df, flattened_json_df], axis=1).drop(columns=['d'])
        
        # 5. 결과 파일 저장
        output_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_parsed.csv")
        df_expanded.to_csv(output_filename, index=False)
        
        return f"Successfully processed '{base_filename}' -> '{os.path.basename(output_filename)}'"

    except Exception as e:
        return f"Error processing '{os.path.basename(input_filename)}': {e}"

def main():
    # ====================================================================
    # (개선) argparse를 사용하여 커맨드라인에서 경로를 입력받습니다.
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