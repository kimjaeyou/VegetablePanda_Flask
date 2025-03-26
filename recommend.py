from flask import Flask, request, jsonify
import pandas as pd
import pymysql
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine


app = Flask(__name__)

@app.route('/recommend/py', methods=['POST'])
def recommend():
    
    data = request.get_json()
    user_seq = data.get('user_seq')
    
    
    db = pymysql.connect(
        host='',     # MySQL Server Address
        port=,     # MySQL Server Port
        user='',      # MySQL username
        passwd='',    # password for MySQL username
        db='finalDB',   
        charset='utf8'
    )

    sql="SELECT rc.review_comment_seq,rc.user_seq, r.review_seq,s.color, ubd.price, rc.score FROM review_comment rc JOIN review r ON rc.review_seq = r.review_seq JOIN user_buy_detail ubd ON rc.user_buy_detail_seq = ubd.user_buy_detail_seq JOIN stock s ON ubd.stock_seq = s.stock_seq;";
    df = pd.read_sql(sql, db)
    db.close() 

    user_item_matrix = df.pivot(index='user_seq', columns='review_seq', values='score').fillna(0)

    item_metadata = df[['review_seq', 'price', 'color']].drop_duplicates().set_index('review_seq')

    scaler = StandardScaler()
    item_metadata['price_scaled'] = scaler.fit_transform(item_metadata[['price']])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
    color_encoded = encoder.fit_transform(item_metadata[['color']])
    color_encoded_df = pd.DataFrame(
        color_encoded,
        columns=[f"color_{cat}" for cat in encoder.categories_[0]],
        index=item_metadata.index
    )

    item_metadata_expanded = pd.concat([item_metadata[['price_scaled']], color_encoded_df], axis=1)

    user_item_matrix_expanded = user_item_matrix.T.join(item_metadata_expanded, how='left').T.fillna(0)

    X = user_item_matrix_expanded.values

    model = TruncatedSVD(n_components=10, n_iter=100, random_state=6)  
    model.fit(X)


    predicted_ratings_matrix = model.inverse_transform(model.transform(X))

    user_id = user_seq
    if user_id in user_item_matrix.index:  # 사용자 ID가 user_item_matrix에 존재하는지 확인
        user_ratings = user_item_matrix.loc[user_id].values  # 해당 사용자의 실제 평점
        predicted_ratings_for_user = predicted_ratings_matrix[user_item_matrix.index.get_loc(user_id)]  # 해당 사용자의 예측 평점

        # 16. 사용자가 평가하지 않은 상품에 대해 예측 평점 출력
        recommended_items = []
        for item_idx, (actual_rating, predicted_rating) in enumerate(zip(user_ratings, predicted_ratings_for_user)):
            if actual_rating == 0:  # 사용자가 평가하지 않은 상품
                review_seq = user_item_matrix.columns[item_idx]  # review_seq를 가져옴
                recommended_items.append((review_seq, predicted_rating))  # 상품 ID와 예측 평점 저장

        # 17. 예측 평점이 높은 순으로 정렬 (상위 5개 추천)
        recommended_items.sort(key=lambda x: x[1], reverse=True)
        top_recommended_items = recommended_items[:5]
        
        
        filtered_recommended_items = [item for item in recommended_items if item[1] >= 3]

        
        top_recommended_items = filtered_recommended_items[:3]

        # NumPy 타입을 Python 기본 타입으로 변환
        top_recommended_items = [
            int(item[0])
            for item in top_recommended_items
            ]
        print(top_recommended_items)

    return jsonify(top_recommended_items)




@app.route('/recommendShop/py', methods=['POST'])
def recommendShop():
    # 클라이언트에서 전달받은 shop_seq
    data = request.get_json()
    shop_seq = data.get('shop_seq')
    if shop_seq is None:
        return jsonify({"message": "shop_seq is required"}), 400
    
    try:
        shop_seq = int(shop_seq)  # 정수형 변환
    except ValueError:
        return jsonify({"message": "Invalid shop_seq format. Must be an integer."}), 400
    
    # SQLAlchemy 엔진 생성
    engine = create_engine('mysql+pymysql://root:jmkh425124@finaldb.c3cgiqia4q8c.ap-northeast-2.rds.amazonaws.com:3306/finalDB')

    # SQL 쿼리 실행
    sql = """
    SELECT s.color, sh.shop_seq, s.product_seq, p.product_name, 
           s.stock_grade_seq, s.stock_organic_seq, s.stock_seq, s.user_seq, 
           p.product_category_seq, sh.price
    FROM stock s
    JOIN product p ON s.product_seq = p.product_seq
    JOIN shop sh ON sh.stock_seq = s.stock_seq;
    """
    try:
        df = pd.read_sql(sql, engine)
    except Exception as e:
        return jsonify({"message": f"Database query failed: {str(e)}"}), 500

    if shop_seq not in df['shop_seq'].values:
        return jsonify({"message": f"Shop with shop_seq={shop_seq} not found."}), 404

    # Step 1: 범주형 변수 One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    categorical_features = encoder.fit_transform(df[['shop_seq', 'product_category_seq', 'stock_grade_seq', 'stock_organic_seq']])

    # Step 2: 가격 데이터 정규화
    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(df[['price']])  # 가격 데이터 정규화

    # Step 3: 범주형 특성 + 가격 특성 결합
    features = np.hstack([categorical_features, price_scaled])

    # Step 4: 코사인 유사도 계산
    cosine_sim = cosine_similarity(features, features)

    # Step 5: 추천 함수 (shop_seq 기준)
    idx = df.index[df['shop_seq'] == shop_seq][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # 자기 자신 제외 Top 5

    # 추천된 shop_seq 반환
    shop_indices = [i[0] for i in sim_scores]
    recommended_shops = df.iloc[shop_indices]['shop_seq'].tolist()
    print(recommended_shops)

    return jsonify(recommended_shops)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
