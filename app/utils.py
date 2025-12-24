"""
共通ユーティリティ関数
"""


def filter_text(text: str, min_length: int = 2, exclude_whitespace_only: bool = True) -> str:
    """基本的なテキストフィルタリング（品質向上用）
    
    Args:
        text: フィルタリング対象のテキスト
        min_length: 最小文字数（これ未満は除外）
        exclude_whitespace_only: 空白のみの文字列を除外するかどうか
        
    Returns:
        フィルタリング後のテキスト（除外される場合は空文字列）
    """
    if not text:
        return ""
    
    # 空白のみのチェック
    if exclude_whitespace_only and not text.strip():
        return ""
    
    # 最小文字数のチェック（空白を除いた文字数）
    if len(text.strip()) < min_length:
        return ""
    
    return text

