import optuna

# Specify the storage URL
storage_url = 'sqlite:///db.sqlite3'

# Create a storage object
storage = optuna.storages.RDBStorage(storage_url)

# Initialize the database schema
optuna.storages._rdb.models.Base.metadata.create_all(storage.engine)

print("Database schema initialized successfully.")
