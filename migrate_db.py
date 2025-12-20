import os
import sys
from app import app, db, Message, Chat, User
from sqlalchemy import text as sqltext

def migrate_database():
    """Safely add missing columns to Message table"""
    with app.app_context():
        try:
            # Get database URL
            db_url = os.getenv("DATABASE_URL", "sqlite:///nexa-ai.db")
            
            if "postgresql" in db_url or "postgres" in db_url:
                # PostgreSQL migration
                with db.engine.connect() as conn:
                    conn.begin()
                    
                    # Check and add missing columns
                    try:
                        conn.execute(sqltext("""
                            ALTER TABLE message 
                            ADD COLUMN imageurl VARCHAR(1000);
                        """))
                        print("✓ Added imageurl column")
                    except Exception as e:
                        if "already exists" in str(e):
                            print("✓ imageurl column already exists")
                        else:
                            print(f"Warning: {e}")
                    
                    try:
                        conn.execute(sqltext("""
                            ALTER TABLE message 
                            ADD COLUMN imagepath VARCHAR(1000);
                        """))
                        print("✓ Added imagepath column")
                    except Exception as e:
                        if "already exists" in str(e):
                            print("✓ imagepath column already exists")
                        else:
                            print(f"Warning: {e}")
                    
                    try:
                        conn.execute(sqltext("""
                            ALTER TABLE message 
                            ADD COLUMN imagedata TEXT;
                        """))
                        print("✓ Added imagedata column")
                    except Exception as e:
                        if "already exists" in str(e):
                            print("✓ imagedata column already exists")
                        else:
                            print(f"Warning: {e}")
                    
                    conn.commit()
            
            else:
                # SQLite migration
                with db.engine.connect() as conn:
                    conn.begin()
                    
                    # For SQLite, we need to recreate the table with new columns
                    try:
                        # Check current schema
                        result = conn.execute(sqltext("PRAGMA table_info(message)"))
                        columns = [row[1] for row in result.fetchall()]
                        
                        # Add missing columns if they don't exist
                        if "imageurl" not in columns:
                            conn.execute(sqltext("""
                                ALTER TABLE message 
                                ADD COLUMN imageurl VARCHAR(1000);
                            """))
                            print("✓ Added imageurl column to SQLite")
                        
                        if "imagepath" not in columns:
                            conn.execute(sqltext("""
                                ALTER TABLE message 
                                ADD COLUMN imagepath VARCHAR(1000);
                            """))
                            print("✓ Added imagepath column to SQLite")
                        
                        if "imagedata" not in columns:
                            conn.execute(sqltext("""
                                ALTER TABLE message 
                                ADD COLUMN imagedata TEXT;
                            """))
                            print("✓ Added imagedata column to SQLite")
                        
                        conn.commit()
                    except Exception as e:
                        print(f"SQLite migration error: {e}")
                        conn.rollback()
                        raise
            
            print("✓ Database migration completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)
