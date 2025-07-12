#!/usr/bin/env python3
"""
Database migration system for the Genomic Pleiotropy Cryptanalysis trial database.
Manages schema updates and data migrations.
"""
import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Callable, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, description: str, 
                 upgrade_func: Callable, downgrade_func: Callable = None):
        self.version = version
        self.description = description
        self.upgrade_func = upgrade_func
        self.downgrade_func = downgrade_func
        self.applied_at = None
    
    def upgrade(self, connection):
        """Apply the migration."""
        self.upgrade_func(connection)
    
    def downgrade(self, connection):
        """Revert the migration."""
        if self.downgrade_func:
            self.downgrade_func(connection)
        else:
            raise NotImplementedError(f"No downgrade defined for migration {self.version}")


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, db_path: str = "trial_database/database/trials.db"):
        self.db_path = Path(db_path)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)
        self.migrations: List[Migration] = []
        
        # Initialize migration tracking table
        self._init_migration_table()
        
        # Register all migrations
        self._register_migrations()
    
    def _init_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
    
    def _register_migrations(self):
        """Register all available migrations."""
        # Migration 001: Add experiment tags
        def upgrade_001(conn):
            conn.execute(text("""
                ALTER TABLE trials ADD COLUMN tags JSON DEFAULT '[]'
            """))
            conn.execute(text("""
                CREATE INDEX idx_trials_tags ON trials(tags)
            """))
        
        def downgrade_001(conn):
            conn.execute(text("""
                DROP INDEX IF EXISTS idx_trials_tags
            """))
            # SQLite doesn't support DROP COLUMN, would need to recreate table
        
        self.migrations.append(Migration(
            "001",
            "Add tags field to trials table",
            upgrade_001,
            downgrade_001
        ))
        
        # Migration 002: Add result metadata
        def upgrade_002(conn):
            conn.execute(text("""
                ALTER TABLE results ADD COLUMN metadata JSON DEFAULT '{}'
            """))
            conn.execute(text("""
                ALTER TABLE results ADD COLUMN computation_time_seconds REAL
            """))
        
        self.migrations.append(Migration(
            "002",
            "Add metadata and computation time to results",
            upgrade_002
        ))
        
        # Migration 003: Add agent capabilities
        def upgrade_003(conn):
            conn.execute(text("""
                ALTER TABLE agents ADD COLUMN capabilities JSON DEFAULT '[]'
            """))
            conn.execute(text("""
                ALTER TABLE agents ADD COLUMN configuration JSON DEFAULT '{}'
            """))
        
        self.migrations.append(Migration(
            "003",
            "Add capabilities and configuration to agents",
            upgrade_003
        ))
        
        # Migration 004: Add trial relationships
        def upgrade_004(conn):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trial_dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id INTEGER NOT NULL,
                    depends_on_trial_id INTEGER NOT NULL,
                    dependency_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trial_id) REFERENCES trials(id) ON DELETE CASCADE,
                    FOREIGN KEY (depends_on_trial_id) REFERENCES trials(id) ON DELETE CASCADE,
                    UNIQUE(trial_id, depends_on_trial_id)
                )
            """))
            conn.execute(text("""
                CREATE INDEX idx_trial_deps_trial ON trial_dependencies(trial_id)
            """))
            conn.execute(text("""
                CREATE INDEX idx_trial_deps_depends ON trial_dependencies(depends_on_trial_id)
            """))
        
        self.migrations.append(Migration(
            "004",
            "Add trial dependency tracking",
            upgrade_004
        ))
        
        # Migration 005: Add audit log
        def upgrade_005(conn):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    record_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('insert', 'update', 'delete')),
                    old_values JSON,
                    new_values JSON,
                    changed_by_agent INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (changed_by_agent) REFERENCES agents(id)
                )
            """))
            conn.execute(text("""
                CREATE INDEX idx_audit_table ON audit_log(table_name)
            """))
            conn.execute(text("""
                CREATE INDEX idx_audit_timestamp ON audit_log(timestamp)
            """))
        
        self.migrations.append(Migration(
            "005",
            "Add audit logging table",
            upgrade_005
        ))
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT version FROM schema_migrations ORDER BY version"
            ))
            return [row[0] for row in result]
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of migrations that haven't been applied yet."""
        applied = self.get_applied_migrations()
        return [m for m in self.migrations if m.version not in applied]
    
    def apply_migration(self, migration: Migration):
        """Apply a single migration."""
        with self.engine.begin() as conn:
            try:
                print(f"Applying migration {migration.version}: {migration.description}")
                
                # Apply the migration
                migration.upgrade(conn)
                
                # Record the migration
                conn.execute(text("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (:version, :description)
                """), {"version": migration.version, "description": migration.description})
                
                print(f"  ✓ Migration {migration.version} applied successfully")
                
            except Exception as e:
                print(f"  ✗ Error applying migration {migration.version}: {e}")
                raise
    
    def migrate(self, target_version: str = None):
        """Apply all pending migrations up to target version."""
        pending = self.get_pending_migrations()
        
        if not pending:
            print("Database is up to date. No migrations to apply.")
            return
        
        print(f"Found {len(pending)} pending migrations")
        
        for migration in pending:
            if target_version and migration.version > target_version:
                break
            self.apply_migration(migration)
        
        print("\nMigration complete!")
    
    def rollback(self, target_version: str):
        """Rollback to a specific version."""
        applied = self.get_applied_migrations()
        
        if target_version not in applied:
            print(f"Target version {target_version} is not in applied migrations")
            return
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for migration in reversed(self.migrations):
            if migration.version not in applied:
                continue
            if migration.version <= target_version:
                break
            to_rollback.append(migration)
        
        if not to_rollback:
            print("No migrations to rollback")
            return
        
        print(f"Rolling back {len(to_rollback)} migrations")
        
        for migration in to_rollback:
            with self.engine.begin() as conn:
                try:
                    print(f"Rolling back migration {migration.version}: {migration.description}")
                    
                    # Apply the downgrade
                    migration.downgrade(conn)
                    
                    # Remove the migration record
                    conn.execute(text("""
                        DELETE FROM schema_migrations WHERE version = :version
                    """), {"version": migration.version})
                    
                    print(f"  ✓ Migration {migration.version} rolled back successfully")
                    
                except Exception as e:
                    print(f"  ✗ Error rolling back migration {migration.version}: {e}")
                    raise
        
        print("\nRollback complete!")
    
    def status(self):
        """Show migration status."""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        
        print("Migration Status")
        print("=" * 50)
        
        print("\nApplied Migrations:")
        if applied:
            with self.engine.connect() as conn:
                for version in applied:
                    result = conn.execute(text("""
                        SELECT description, applied_at 
                        FROM schema_migrations 
                        WHERE version = :version
                    """), {"version": version})
                    row = result.fetchone()
                    print(f"  ✓ {version}: {row[0]} (applied at {row[1]})")
        else:
            print("  No migrations applied yet")
        
        print("\nPending Migrations:")
        if pending:
            for migration in pending:
                print(f"  - {migration.version}: {migration.description}")
        else:
            print("  No pending migrations")
        
        print("\nCurrent schema version:", applied[-1] if applied else "000")


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage database migrations")
    parser.add_argument(
        "--db-path",
        default="trial_database/database/trials.db",
        help="Path to the database file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Migration commands")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Apply pending migrations")
    migrate_parser.add_argument(
        "--target",
        help="Target version to migrate to"
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument(
        "target",
        help="Target version to rollback to"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize migration manager
    manager = MigrationManager(args.db_path)
    
    # Execute command
    if args.command == "migrate":
        manager.migrate(args.target)
    elif args.command == "rollback":
        manager.rollback(args.target)
    elif args.command == "status":
        manager.status()


if __name__ == "__main__":
    main()