from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
from datetime import datetime
import uuid

from text_summarizer import summarize_text, summarize_url
from models import db, Summary

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Configure the database
# Use SQLite locally, but on Vercel we'll use an in-memory database
# This is because Vercel functions are stateless and can't write to disk
is_vercel = os.environ.get('VERCEL') == '1'
if is_vercel:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///text_summarizer.db"
    
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database
db.init_app(app)

# Create all tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize the provided content."""
    content_type = request.form.get('content_type', 'text')
    content = request.form.get('content', '').strip()
    
    if not content:
        return render_template('index.html', error="Please provide text or a URL to summarize.")
    
    try:
        # Track the request in session history
        if 'history' not in session:
            session['history'] = []
        
        session_id = str(uuid.uuid4())
        
        # Generate summary based on content type
        if content_type == 'url':
            result = summarize_url(content)
        else:
            result = summarize_text(content)
        
        # Save the summary to the database
        with app.app_context():
            summary = Summary(
                content_type=content_type,
                original_content=content,
                summary_text=result['summary'],
                original_length=result['original_length'],
                summary_length=result['summary_length'],
                reduction_percentage=result['reduction_percentage']
            )
            db.session.add(summary)
            db.session.commit()
            
            # Add to session history (stores only ID to keep session light)
            session['history'].insert(0, {'id': summary.id, 'timestamp': datetime.now().isoformat()})
            if len(session['history']) > 10:  # Limit to last 10 entries
                session['history'] = session['history'][:10]
            session.modified = True
        
        return render_template('results.html', result=result, content=content, content_type=content_type)
    
    except Exception as e:
        return render_template('index.html', error=f"Error processing your request: {str(e)}")

@app.route('/history')
def history():
    """Display the history of summarized content."""
    if 'history' not in session or not session['history']:
        return render_template('history.html', history=[])
    
    # Retrieve full summary objects from database
    with app.app_context():
        # Get IDs from session history
        ids = [entry['id'] for entry in session['history']]
        # Query the database for these summaries
        summaries = Summary.query.filter(Summary.id.in_(ids)).all()
        
        # Sort summaries to match session history order
        id_to_summary = {summary.id: summary for summary in summaries}
        ordered_summaries = [id_to_summary.get(id) for id in ids]
        # Filter out any None values (in case some summaries were deleted)
        ordered_summaries = [s for s in ordered_summaries if s is not None]
        
    return render_template('history.html', history=ordered_summaries)

@app.route('/about')
def about():
    """Display information about the text summarizer."""
    return render_template('about.html')

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the summary history."""
    if 'history' in session:
        session.pop('history')
    return redirect(url_for('history'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)