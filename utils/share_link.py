def get_share_links(result_label: str) -> dict:
    """
    SNS 공유를 위한 예시 링크를 반환합니다.
    실제 서비스별 공유 API나 쿼리 파라미터를 추가로 설정해줘야 합니다.
    """
    share_text = f"나는 {result_label} 판정을 받았어요!"
    twitter_url = f"https://twitter.com/intent/tweet?text={share_text}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u=http://example.com&quote={share_text}"

    return {
        "twitter": twitter_url,
        "facebook": facebook_url
    }
