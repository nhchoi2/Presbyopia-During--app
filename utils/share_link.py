def get_share_links(label: str) -> dict:
    """
    SNS(예: 트위터, 페이스북) 공유 링크를 단순히 생성한 예시입니다.
    실제로는 서비스마다 파라미터 설정이 달라질 수 있습니다.
    """
    text = f"나는 지금 {label[2]} 판정을 받았어요!"
    twitter_url = f"https://twitter.com/intent/tweet?text={text}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u=https://example.com&quote={text}"
    naver_url = f"https://share.naver.com/web/shareView.nhn?url={short_url}&title=동안 vs 노안 테스트 결과!",

    return {
        "twitter": twitter_url,
        "facebook": facebook_url,
        "네이버 블로그" : naver_url
    }
