from sqlalchemy.schema import CreateTable
from sqlalchemy import create_engine
from models import Base
import config

def generate_sql_schema():
    # We create the engine right here so SQLAlchemy knows we are using SQLite
    engine = create_engine(config.DATABASE_URI)
    
    schema_path = config.PROJECT_ROOT / "schema.sql"
    with open(schema_path, "w") as f:
        for table in Base.metadata.sorted_tables:
            # Generate the CREATE TABLE string for SQLite
            create_statement = str(CreateTable(table).compile(engine))
            f.write(create_statement + ";\n\n")
            
    print(f"Schema successfully exported to {schema_path}")

if __name__ == "__main__":
    generate_sql_schema()