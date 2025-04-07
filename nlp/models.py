from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class Summary(db.Model):
    """Model representing a text summary."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    content_type = db.Column(db.String(10), nullable=False)  # 'text' or 'url'
    original_content = db.Column(db.Text, nullable=False)
    summary_text = db.Column(db.Text, nullable=False)
    original_length = db.Column(db.Integer, nullable=False)
    summary_length = db.Column(db.Integer, nullable=False)
    reduction_percentage = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<Summary {self.id}: {self.timestamp}>'
    
    def to_dict(self):
        """Convert summary to a dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'content_type': self.content_type,
            'original_content': self.original_content,
            'summary_text': self.summary_text,
            'original_length': self.original_length,
            'summary_length': self.summary_length,
            'reduction_percentage': self.reduction_percentage
        }