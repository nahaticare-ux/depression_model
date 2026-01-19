import streamlit as st
import Orange
import pickle
import warnings

# 1. 환경 차이로 인한 UserWarning 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 2. 모델 불러오기 함수
@st.cache_resource
def load_model():
    # 깃허브에 올린 파일명이 정확히 depression_model.pkcls 여야 합니다.
    model_path = "depression_model.pkcls"
    with open(model_path, "rb") as f:
        return pickle.load(f)

# 모델 로드 (파일이 없을 경우 대비)
try:
    model = load_model()
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다. 파일명을 확인해주세요: {e}")
    st.stop() # 모델이 없으면 실행 중단

# 3. 페이지 디자인
st.set_page_config(page_title="마음기상청", page_icon="☁️")
st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("인공지능(랜덤 포레스트)이 당신의 일상을 분석하여 마음의 날씨를 알려드립니다.")

# 4. 사용자 입력 받기
st.divider()
stress = st.slider("🔥 오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("😴 어제 수면 시간 (0~24시간)", 0.0, 24.0, 7.0)
social = st.number_input("📱 SNS 사용 시간 (0~24시간)", 0.0, 24.0, 2.0)

# 5. 분석하기 버튼 클릭 시 작동
if st.button("마음 날씨 예보하기"):
    try:
        # [핵심] 모델이 요구하는 9개의 칸(Domain)을 정확한 순서로 채웁니다.
        # 순서: Age(21), Gender(1), Sleep, Study(5), Social, Physical(3), Stress, Target(0), Meta(0)
        input_data = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        
        # Orange 전용 데이터 인스턴스 생성
        inst = Orange.data.Instance(model.domain, input_data)
        
        # 예측 수행
        prediction = model(inst)
        
        # [해결] 결과값 상자 열기 (Scalar Variable 오류 완벽 방지)
        if hasattr(prediction, "__len__"):
            pred_value = int(prediction[0])
        else:
            pred_value = int(prediction)
        
        # 확률값 추출 (probs[1]은 우울증 위험도)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        risk_prob = float(probs[1]) * 100

        # 6. 결과 출력
        st.divider()
        if pred_value == 1:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_prob:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_prob:.1f}%)")

    except Exception as e:
        # 가장 빈번한 'only length-1 arrays' 오류를 여기서 마지막으로 잡아줍니다.
        st.error(f"분석 엔진 오류: {e}")
