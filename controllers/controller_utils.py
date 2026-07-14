"""
Shared utilities for API controllers
"""
from flask import jsonify


def success_response(message, data, status_code=200):
    """Standard success response"""
    return jsonify({
        'status': 'success',
        'message': message,
        'data': data,
        'status_code': status_code
    }), status_code


def error_response(message, status_code=500, data=None):
    """Standard error response"""
    return jsonify({
        'status': 'error',
        'message': message,
        'data': data or {},
        'status_code': status_code
    }), status_code
