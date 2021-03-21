"""create account table

Revision ID: 4fb4b8a53853
Revises: 
Create Date: 2021-02-17 15:09:19.958908

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4fb4b8a53853'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('User', sa.Column('zipcode', String(6), nullable=True)


def downgrade():
    pass
