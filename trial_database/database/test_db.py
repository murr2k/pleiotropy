#!/usr/bin/env python3
"""
Test script to verify database setup without external dependencies.
Uses only sqlite3 from the standard library.
"""
import sqlite3
import json
from pathlib import Path


def test_database():
    """Test database creation and basic operations."""
    db_path = Path("trial_database/database/trials.db")
    
    print(f"Testing database at: {db_path}")
    print("=" * 50)
    
    # Check if database exists
    if not db_path.exists():
        print("❌ Database file does not exist!")
        print("   Run: python3 trial_database/database/init_db.py --seed")
        return False
    
    print("✓ Database file exists")
    
    # Connect to database
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        print("✓ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Test each table
    tables = ['agents', 'trials', 'results', 'progress']
    
    for table in tables:
        try:
            # Check if table exists
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"✓ Table '{table}' exists with {count} records")
            
            # Show sample data
            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            rows = cursor.fetchall()
            if rows:
                print(f"  Sample data from {table}:")
                for row in rows:
                    # Convert row to dict for display
                    data = dict(row)
                    # Truncate long fields
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 50:
                            data[key] = value[:47] + "..."
                    print(f"    {data}")
            
        except Exception as e:
            print(f"❌ Error checking table '{table}': {e}")
            return False
    
    # Test a join query
    print("\n✓ Testing join query...")
    try:
        cursor.execute("""
            SELECT t.name as trial_name, r.confidence_score, a.name as agent_name
            FROM trials t
            JOIN results r ON t.id = r.trial_id
            JOIN agents a ON r.agent_id = a.id
            ORDER BY r.confidence_score DESC
            LIMIT 3
        """)
        
        print("  Top results by confidence:")
        for row in cursor.fetchall():
            print(f"    - {row['trial_name']}: {row['confidence_score']} (by {row['agent_name']})")
    
    except Exception as e:
        print(f"❌ Join query failed: {e}")
    
    # Test JSON field access (SQLite 3.9+)
    print("\n✓ Testing JSON field access...")
    try:
        cursor.execute("""
            SELECT name, json_extract(parameters, '$.window_size') as window_size
            FROM trials
            WHERE json_extract(parameters, '$.window_size') IS NOT NULL
            LIMIT 3
        """)
        
        print("  Trials with window_size parameter:")
        for row in cursor.fetchall():
            print(f"    - {row['name']}: window_size = {row['window_size']}")
    
    except Exception as e:
        print(f"  ⚠️  JSON query not supported (SQLite version may be too old): {e}")
    
    # Close connection
    conn.close()
    
    print("\n" + "=" * 50)
    print("✓ All basic database tests passed!")
    return True


def show_schema():
    """Display the database schema."""
    schema_path = Path("trial_database/database/schema.sql")
    
    if schema_path.exists():
        print("\nDatabase Schema:")
        print("=" * 50)
        with open(schema_path, 'r') as f:
            # Show first 50 lines
            lines = f.readlines()[:50]
            print(''.join(lines))
            if len(lines) == 50:
                print("... (truncated)")


if __name__ == "__main__":
    # Run tests
    success = test_database()
    
    if not success:
        print("\n⚠️  Database tests failed!")
        print("To initialize the database, first install dependencies:")
        print("  pip install -r trial_database/requirements.txt")
        print("Then run:")
        print("  python3 trial_database/database/init_db.py --seed")
    else:
        # Show schema if tests pass
        show_schema()