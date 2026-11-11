#!/usr/bin/env python3
"""One-time desktop OAuth consent for owner-requested Gmail reads and draft creation; stdlib only."""
import argparse
import base64
import getpass
import hashlib
import json
import secrets
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCOPE = 'https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.compose'
TOKEN_URL = 'https://oauth2.googleapis.com/token'


def consent_url(client_id, redirect_uri, state, challenge):
    return 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode({
        'client_id': client_id, 'redirect_uri': redirect_uri, 'response_type': 'code',
        'scope': SCOPE, 'access_type': 'offline', 'prompt': 'consent', 'state': state,
        'code_challenge': challenge, 'code_challenge_method': 'S256',
        'include_granted_scopes': 'false'})


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Unexpected token endpoint redirect')


def exchange_code(client_id, client_secret, code, redirect_uri, verifier, *, open_url=None):
    request = Request(TOKEN_URL, data=urlencode({
        'client_id': client_id, 'client_secret': client_secret, 'code': code,
        'redirect_uri': redirect_uri, 'grant_type': 'authorization_code',
        'code_verifier': verifier}).encode(), method='POST',
        headers={'Content-Type': 'application/x-www-form-urlencoded'})
    opener = open_url or build_opener(NoRedirect()).open
    with opener(request, timeout=20) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError('Invalid token response')
    if not isinstance(payload.get('scope'), str) or set(payload['scope'].split()) != set(SCOPE.split()):
        raise ValueError('Unexpected granted scope')
    token = payload.get('refresh_token')
    if not isinstance(token, str) or not token.strip():
        raise ValueError('No refresh token returned')
    return token


def callback_handler(state, result):
    class Callback(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # The URL contains a code: never log it.

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            valid = (parsed.path == '/' and query.get('state') == [state]
                     and (len(query.get('code', [])) == 1 or 'error' in query))
            if valid:
                result.update(code=query.get('code', [None])[0], error='error' in query)
            self.send_response(200 if valid else 400)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(b'Return to the terminal.' if valid else b'Invalid callback.')
    return Callback


def env_lines(client_id, client_secret, refresh_token):
    # Google credentials use a restricted alphabet. Reject dotenv metacharacters;
    # never produce injected env lines or ${...} interpolation.
    import re
    values = (client_id, client_secret, refresh_token)
    if any(not re.fullmatch(r'[A-Za-z0-9._~+/-]+', value) for value in values):
        raise ValueError('Unexpected credential format')
    return '\n'.join(f'{key}={value}' for key, value in zip(
        ('GMAIL_DRAFTS_CLIENT_ID', 'GMAIL_DRAFTS_CLIENT_SECRET', 'GMAIL_DRAFTS_REFRESH_TOKEN'), values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        'Run once on a machine with a browser to authorize Gmail readonly and compose access. '
        'The scope can create drafts and send, but the brain code only creates drafts. '
        'Uses a Desktop app OAuth client and a random 127.0.0.1 loopback port. '
        'Prints only three credential env lines to stdout for manual pasting into the brain .env; '
        'writes no files. Set GMAIL_READ=1 and/or GMAIL_DRAFTS=1; set GMAIL_DRAFTS_FROM to your address.'))
    parser.add_argument('--client-json', metavar='PATH', help='Google Desktop app client JSON; otherwise prompt')
    args = parser.parse_args(argv)
    try:
        if args.client_json:
            with open(args.client_json, encoding='utf-8') as source:
                credentials = json.load(source)['installed']
            client_id, client_secret = credentials['client_id'], credentials['client_secret']
        else:
            client_id = getpass.getpass('OAuth client id: ', stream=sys.stderr).strip()
            client_secret = getpass.getpass('OAuth client secret: ', stream=sys.stderr).strip()
        env_lines(client_id, client_secret, 'validation-placeholder')
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        result = {}
        with HTTPServer(('127.0.0.1', 0), callback_handler(state, result)) as server:
            server.timeout = 1
            redirect_uri = f'http://127.0.0.1:{server.server_port}/'
            if not webbrowser.open(consent_url(client_id, redirect_uri, state, challenge)):
                raise ValueError('Browser unavailable')
            deadline = time.monotonic() + 300
            while not result and time.monotonic() < deadline:
                server.handle_request()
        if result.get('error') or not result.get('code'):
            raise ValueError('Consent failed or timed out')
        refresh_token = exchange_code(client_id, client_secret, result['code'], redirect_uri, verifier)
        output = env_lines(client_id, client_secret, refresh_token)
    except (Exception, KeyboardInterrupt):
        print('Sign-in failed; no credentials were printed. Check the Desktop client setup and retry.', file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
