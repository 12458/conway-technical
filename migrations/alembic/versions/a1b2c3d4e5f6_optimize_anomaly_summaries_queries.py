"""Optimize anomaly_summaries queries

Revision ID: a1b2c3d4e5f6
Revises: 57b0c6d72c46
Create Date: 2025-11-01 21:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "57b0c6d72c46"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add composite index for efficient filtering by created_at and severity
    # This supports queries like: WHERE created_at > X ORDER BY created_at DESC
    # and queries that also filter by severity
    op.create_index(
        "ix_anomaly_summaries_created_at_severity",
        "anomaly_summaries",
        [sa.text("created_at DESC"), "severity"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_anomaly_summaries_created_at_severity",
        table_name="anomaly_summaries",
    )
