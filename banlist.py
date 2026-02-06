import requests

def get_as_ipv4_prefixes(asn):
    """
    获取指定 ASN 的所有 IPv4 宣告段
    """
    asn = asn.upper()
    if not asn.startswith('AS'):
        asn = f'AS{asn}'
        
    url = f"https://stat.ripe.net/data/announced-prefixes/data.json?resource={asn}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        prefixes = data.get('data', {}).get('prefixes', [])
        ipv4_prefixes = []
        
        for p in prefixes:
            prefix = p.get('prefix')
            # 过滤 IPv4 (包含 '.' 的是 IPv4)
            if '.' in prefix:
                ipv4_prefixes.append(prefix)
        
        # 去重并排序
        return sorted(list(set(ipv4_prefixes)))
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

if __name__ == "__main__":
    target_asn = "AS138997"
    results = get_as_ipv4_prefixes(target_asn)
    
    if results:
        print(f"# AS{target_asn} IPv4 CIDR List:")
        for cidr in results:
            print(cidr)
    else:
        print("No prefixes found or error occurred.")