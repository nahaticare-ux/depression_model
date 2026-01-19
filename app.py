import streamlit as st
import Orange
import pickle
import warnings

# 1. 지저분한 환경 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 2. 모델 불러오기
@st.cache_resource
def load_model():
    # 파일명이 depression_model.pkcls 인지 확인하세요!
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

# 3. 페이지 디자인
st.set_page_config(page_title="마음기상청", page_icon="☁️")
st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("인공지능(랜덤 포레스트)이 당신의 일상을 분석하여 마음의 날씨를 알려드립니다.")

# 4. 사용자 입력 받기
st.divider()
stress = st.slider("🔥 오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("😴 어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("📱 SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 5. 분석하기 버튼
if st.button("마음 날씨 예보하기"):
    try:
        # [데이터 규격] 모델이 요구하는 9개의 칸을 정확히 채웁니다.
        # 순서: Age(21), Gender(1), Sleep, Study(5), Social, Physical(3), Stress, Target(0), Meta(0)
        input_list = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        
        inst = Orange.data.Instance(model.domain, input_list)
        
        # [해결] 랜덤 포레스트 전용 결과 추출 로직
        # 현재 오류(scalar variable)는 prediction을 리스트로 다루려 해서 발생합니다.
        prediction = model(inst)
        
        # 확률값 추출 (probs[1]은 우울증일 확률)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        risk_prob = float(probs[1]) * 100

        # 6. 결과 출력
        st.divider()
        # prediction이 숫자인지 배열인지에 상관없이 안전하게 처리합니다.
        result_class = int(prediction[0]) if hasattr(prediction, "__len__") else int(prediction)

        if result_class == 1:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_prob:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_prob:.1f}%)")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
