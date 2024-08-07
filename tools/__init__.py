import warnings
from .cleansing import *
from .calculate import *

def _import_load_data():
    # 포맷이 개선된 경고 메시지
    warning_message = (
        "\n\n"
        "사용법 안내: `load_data` 함수 사용 시, 인자는 다음 순서로 전달해야 합니다:\n"
        "\n"
        "1. participant 테이블 \n"
        "2. position 테이블 \n"
        "3. match_info 테이블 \n"
        "4. objectives 테이블 \n"
        "\n"
        "정확한 인자 순서를 지켜 주세요. \n"
        "예시) participant, position, match_info, objectives = load_data()\n\n"
        "load_data 시, Dask 데이터프레임으로 추출하려면 to_pandas=False 인자를 추가해주세요 \n"
    )
    warnings.warn(warning_message, UserWarning)

# _import_load_data 함수 호출
_import_load_data()