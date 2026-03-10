

def get_community_overview(community_id):
    
    # 1. Total devices
    q1 = f"""
        SELECT COUNT(d.device_uuid) AS total_devices
        FROM device d
        JOIN space s ON d.space_uuid = s.space_uuid
        WHERE s.community_id = '{community_id}';
    """
    total_devices = run_sql(q1)[0]["total_devices"]

    # 2. Device count per space
    q2 = f"""
        SELECT 
            s.space_uuid,
            s.space_name,
            COUNT(d.device_uuid) AS device_count
        FROM space s
        LEFT JOIN device d ON d.space_uuid = s.space_uuid
        WHERE s.community_id = '{community_id}'
        GROUP BY s.space_uuid
        ORDER BY device_count DESC;
    """
    devices_per_space = run_sql(q2)

    # 3. Subspace count per space
    q3 = f"""
        SELECT 
            s.space_uuid,
            s.space_name,
            COUNT(sb.subspace_uuid) AS subspace_count
        FROM space s
        LEFT JOIN subspace sb ON sb.space_uuid = s.space_uuid
        WHERE s.community_id = '{community_id}'
        GROUP BY s.space_uuid;
    """
    subspaces_per_space = run_sql(q3)

    return {
        "total_devices": total_devices,
        "devices_per_space": devices_per_space,
        "subspaces_per_space": subspaces_per_space
    }
