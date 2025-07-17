class Storage:
    def __init__(self, storage_type: str, **kwargs):
        self.storage_type = storage_type
        if storage_type == "s3":
            self.db_name = kwargs.get("storage.db")
            self.db_user = kwargs.get("username")
            self.db_password = kwargs.get("password")

        if storage_type == "disk":
            self.storage_path = kwargs.get('storage_path')
