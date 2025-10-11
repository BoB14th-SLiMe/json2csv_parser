import pandas as pd
import orjson as json  # 기본 json보다 훨씬 빠른 orjson 라이브러리 사용
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def flatten_json(y):
    """중첩된 JSON 객체를 재귀적으로 펼칩니다."""
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def process_file(input_filename):
    """CSV 파일 하나를 읽고, 파싱하여, output 폴더에 저장하는 함수 (단일 작업 단위)"""
    try:
        output_dir = 'output'
        print(f"Processing '{input_filename}'...")

        # 1. CSV 파일 읽기
        df = pd.read_csv(input_filename)

        # 2. 'd' 컬럼이 비어있는 행 제거
        df.dropna(subset=['d'], inplace=True)
        if df.empty:
            print(f"'{input_filename}' has no data to process after dropping NaNs.")
            return

        # 3. orjson을 사용한 빠른 JSON 파싱 및 펼치기
        # .strip('"').replace('""', '"') 부분은 CSV 형식에 따라 필요할 수 있습니다.
        # orjson은 bytes를 선호하므로, 먼저 문자열을 인코딩합니다.
        parsed_json_series = df['d'].apply(lambda x: flatten_json(json.loads(x.strip('"').replace('""', '"'))))

        # 4. 펼친 JSON 데이터를 새로운 데이터프레임으로 변환
        flattened_json_df = pd.DataFrame(parsed_json_series.tolist(), index=df.index)

        # 5. 기존 데이터프레임과 결합 후, 새 파일로 저장
        df_expanded = pd.concat([df.drop(columns=['d']), flattened_json_df], axis=1)
        
        base_filename = os.path.basename(input_filename)
        output_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_parsed.csv")
        df_expanded.to_csv(output_filename, index=False)
        
        return f"Successfully processed '{input_filename}' -> '{output_filename}'"

    except Exception as e:
        return f"Error processing '{input_filename}': {e}"

# ====================================================================
# 여기가 멀티프로세싱이 적용된 메인 실행 부분입니다.
# ====================================================================
if __name__ == "__main__":
    data_dir = 'data'
    output_dir = 'output'
    
    # output 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # data 폴더에서 처리할 csv 파일 목록 가져오기
    files_to_process = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.csv') and '_parsed' not in f
    ]
    
    if not files_to_process:
        print(f"No .csv files to process in '{data_dir}'.")
    else:
        # 시스템의 모든 CPU 코어를 사용
        num_workers = multiprocessing.cpu_count()
        print(f"Starting ProcessPoolExecutor with {num_workers} workers.")
        
        # ProcessPoolExecutor를 사용하여 여러 파일을 병렬로 처리
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 각 파일 처리 작업을 executor에 제출(submit)
            futures = [executor.submit(process_file, filename) for filename in files_to_process]
            
            # 모든 작업이 완료된 후 결과 출력
            for future in futures:
                print(future.result())
        
        print("All processing finished.")