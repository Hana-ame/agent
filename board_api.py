import requests
import time


def request_board(bid=666, timeout=15, retries=3, retry_delay=2):
    """获取 Board 指定帖子列表，支持重试和超时。

    Args:
        bid: Board ID，默认 666
        timeout: 单次请求超时秒数，默认 15
        retries: 最大重试次数，默认 3
        retry_delay: 重试间隔秒数，默认 2

    Returns:
        str 成功时返回 response.text
        None 全部重试失败后返回 None
    """
    url = f"https://vps.moonchan.xyz/api/v2/?bid={bid}&tid=0&pn=0"

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            print(f"[BoardAPI] 超时 (attempt {attempt}/{retries})")
        except requests.exceptions.ConnectionError as e:
            print(f"[BoardAPI] 连接错误 (attempt {attempt}/{retries}): {e}")
        except requests.exceptions.HTTPError as e:
            print(f"[BoardAPI] HTTP 错误 (attempt {attempt}/{retries}): {e}")
        except requests.exceptions.RequestException as e:
            print(f"[BoardAPI] 请求失败 (attempt {attempt}/{retries}): {e}")

        if attempt < retries:
            time.sleep(retry_delay * attempt)

    raise RuntimeError(f"[BoardAPI] 请求 Board {bid} 失败，已重试 {retries} 次")
