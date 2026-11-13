"""Validated cart links, with optional owner-only Discord delivery.

D39 (2026-11-12): owner-approved exception to the D28 guard. Only validated
store IDs and integer quantities enter a fixed message to his configured DM;
no page text, arbitrary destination, login, or purchase crosses this boundary.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal
from urllib.parse import urlencode, urlsplit, urlunsplit

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, BeforeValidator, ConfigDict, StrictBool, StrictInt, StrictStr

from aerys_v2.approvals import owner_id_for
from aerys_v2.hands_gate import OWNER_ASK, OWNER_TEXT, OWNER_VOICE, asker
from aerys_v2.state import identity_from_config, is_lens_surface, is_voice_turn

_HOSTS = {'amazon': {'amazon.com', 'www.amazon.com', 'smile.amazon.com'},
          'walmart': {'walmart.com', 'www.walmart.com'}}
_SHORT = {'a.co', 'amzn.to', 'walmrt.us', 'amzn.com', 'walm.art'}
_IDS = {'amazon': r'[A-Z0-9]{10}', 'walmart': r'[0-9]{5,15}'}


# The schema shows the model both names; a capitalised "Walmart" is lowercased before it's checked.
StoreName = Annotated[Literal['amazon', 'walmart'],
                      BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v)]


class CartItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    product: StrictStr
    quantity: StrictInt = 1


def build_cart_link(store: str, items: list[dict], kind: str = 'cart') -> str:
    """Return a canonical URL (or, for kind='product_page', one canonical product page per line)
    or a refusal; never retain unvalidated URL parts."""
    if kind not in ('cart', 'product_page'):
        return "Refused: kind must be cart or product_page."
    # The live probe (11/12) sent "Walmart" and burned a refusal on capitalisation.
    store = store.strip().lower() if isinstance(store, str) else store
    if not isinstance(store, str) or store not in _HOSTS:
        return 'Refused: store must be amazon or walmart.'
    if not isinstance(items, list) or not 1 <= len(items) <= 10:
        return 'Refused: supply 1–10 items.'
    validated = []
    for index, item in enumerate(items, 1):
        prefix = f'Refused: item {index}: '
        if not isinstance(item, dict) or not isinstance(item.get('product'), str):
            return prefix + 'provide a product page URL or product id.'
        quantity = item.get('quantity', 1)
        if type(quantity) is not int or not 1 <= quantity <= 20:
            return prefix + 'quantity must be an integer from 1 to 20.'
        product = item['product'].strip()
        candidate = product.upper() if store == 'amazon' else product
        # ASCII first: "ß".upper() is "SS", which would turn a non-id into a valid-looking ASIN.
        if not (product.isascii() and re.fullmatch(_IDS[store], candidate)):
            try:
                parsed = urlsplit(product)
                if parsed.hostname in _SHORT:
                    return prefix + 'short link: open it and give me the full product page.'
                if (parsed.scheme not in ('http', 'https') or parsed.hostname not in _HOSTS[store]
                        or parsed.username is not None or parsed.password is not None or parsed.port is not None):
                    return prefix + 'use a full product page from the selected store; do not mix stores.'
                # Dot segments ("/dp/<id>/../../s") resolve away from the product page.
                if any(seg.lower() in ('.', '..', '%2e', '%2e%2e', '.%2e', '%2e.')
                       for seg in parsed.path.split('/')):
                    return prefix + 'not a product page or valid product id; search/category pages are not accepted.'
                if store == 'amazon':
                    # An optional slug before, an optional ref segment after, nothing else.
                    match = re.fullmatch(
                        r'/(?:[^/]+/)?(?:dp|gp/product|gp/aw/d)/([A-Za-z0-9]{10})(?:/[^/]*)?/?', parsed.path)
                else:
                    match = re.fullmatch(r'/ip/(?:[^/]+/)?([0-9]{5,15})/?', parsed.path)
                if match is None:
                    return prefix + 'not a product page or valid product id; search/category pages are not accepted.'
                candidate = match.group(1).upper()
            except ValueError:
                return prefix + 'malformed product URL.'
        validated.append((candidate, quantity))
    if kind == 'product_page':
        # Amazon's cart link silently skips items sold by other sellers (11/12 test tap): the
        # product page lets him press Add to Cart himself. Quantity is his to set there.
        path = '/dp/{}' if store == 'amazon' else '/ip/{}'
        host = 'www.amazon.com' if store == 'amazon' else 'www.walmart.com'
        return '\n'.join(urlunsplit(('https', host, path.format(pid), '', '')) for pid, _ in validated)
    if store == 'amazon':
        params = [(key, value) for i, (pid, qty) in enumerate(validated, 1)
                  for key, value in ((f'ASIN.{i}', pid), (f'Quantity.{i}', qty))]
        return urlunsplit(('https', 'www.amazon.com', '/gp/aws/cart/add.html', urlencode(params), ''))
    query = urlencode({'items': ','.join(f'{pid}_{qty}' for pid, qty in validated)}, safe=',')
    return urlunsplit(('https', 'www.walmart.com', '/sc/cart/addToCart', query, ''))


def _validation_refusal(error) -> str:
    # Do not echo Pydantic's input values: these may contain arbitrary web text.
    first = error.errors()[0]
    loc = first.get('loc', ())
    if len(loc) > 1 and loc[0] == 'items' and isinstance(loc[1], int):
        return f'Refused: item {loc[1] + 1}: invalid product or quantity; use a string product and integer quantity 1–20.'
    return 'Refused: invalid arguments; supply store, 1–10 product items, and an optional boolean send flag.'


def build_cart_link_tool(settings, *, send=None):
    """Inject delivery for offline tests; resolve his destination at send time."""
    @tool
    def add_to_cart_link(store: StoreName, items: list[CartItem], runtime: ToolRuntime,
                         send_to_his_discord: StrictBool = False,
                         kind: Literal['cart', 'product_page'] = 'cart') -> str:
        """Build an Amazon or Walmart cart link from exact product URLs or IDs.

        Supply 1–10 items with product and quantity (default 1, maximum 20).
        Find the exact product first; never invent a cart URL. Amazon's cart link
        only adds items sold by Amazon.com itself: when the listing shows another
        seller, or you can't tell, use kind='product_page' (his tap opens the item
        and he presses Add to Cart). Walmart's cart link works as is. Verified owner
        voice and glasses turns send to his Discord automatically; on text set
        send_to_his_discord only when asked. Nothing is bought.
        """
        raw = [item.model_dump() if isinstance(item, CartItem) else item for item in items]
        link = build_cart_link(store, raw, kind)
        if link.startswith('Refused:'):
            return link
        store = store.strip().lower()
        label = 'Amazon' if store == 'amazon' else 'Walmart'
        count = len(raw)
        noun = 'item' if count == 1 else 'items'
        if kind == 'product_page':
            result = (f'{label} product {"page" if count == 1 else "pages"} ({count} {noun}):\n{link}\n'
                      'Give him these links exactly as written. Each opens the item; he presses Add to Cart.')
            message = f'For your {label} cart, tap Add to Cart on {"it" if count == 1 else "each"}:\n{link}\n-Aerys'
        else:
            result = (f'{label} cart ({count} {noun}):\n{link}\n'
                      'Give him this link exactly as written. His tap adds it to his own cart; nothing is bought.')
            message = f'Add to your {label} cart ({count} {noun}):\n{link}\n-Aerys'
        try:
            config = runtime.config
            identity = identity_from_config(config)
            thread = (config.get('configurable') or {}).get('thread_id')
            requested = send_to_his_discord or is_voice_turn(identity, thread) or is_lens_surface(identity)
            if not requested:
                return result
            who = asker(identity, thread, settings.owner_person_id)
            # Kael's line is his id too, but only he asks for himself (send_message refuses it the same way).
            if who not in (OWNER_TEXT, OWNER_VOICE, OWNER_ASK) or str(thread or '').startswith('kael:'):
                return result + '\nNot sent: it could not be sent to him because the asker is not the verified owner.'
            if send is None or not settings.discord_bot_token:
                return result + '\nNot sent: Discord sending is not available.'
            destination = owner_id_for(settings)()
            if not destination:
                return result + '\nNot sent: his Discord destination is not available.'
            send(settings.discord_bot_token.get_secret_value(), str(destination), message)
            return result + '\nSent the link to his Discord DM.'
        except Exception:
            return result + '\nNot sent: Discord delivery could not be confirmed. Do not claim it was sent.'

    add_to_cart_link.handle_validation_error = _validation_refusal
    return add_to_cart_link
