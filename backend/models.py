from datetime import datetime

def user_helper(user) -> dict:
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "full_name": user.get("full_name"),
        "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
    }

def detection_helper(d) -> dict:
    return {
        "id": str(d["_id"]),
        "user_id": str(d["user_id"]),
        "image_path": d.get("image_path"),
        "result_json": d["result_json"],
        "created_at": d["created_at"].isoformat(),
    }

