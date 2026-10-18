import calendar
import json
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TZ = ZoneInfo('Asia/Kolkata')
ROWS = []
RATES = {}
PRODUCTS = []
READY = False


class Question(BaseModel):
    question: str


def dec(value):
    return Decimal(str(value))


def money(value):
    return float(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def dt(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


async def load_data():
    global ROWS, RATES, PRODUCTS, READY
    READY = False
    url = os.environ.get('LEDGER_API_URL')
    if not url:
        raise RuntimeError('Set LEDGER_API_URL to the full root questionData URL')

    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
        root = (await client.get(url)).raise_for_status().json()
        links = root['links']
        rates_response = await client.get(links['rates'])
        export_response = await client.get(links['export'])
        rates = rates_response.raise_for_status().json()['usd_per_unit']
        exported = export_response.raise_for_status().text

    latest = {}
    for line in exported.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        old = latest.get(row['id'])
        if old is None or dt(row['updated_at']) > dt(old['updated_at']):
            latest[row['id']] = row

    converted = []
    for raw in latest.values():
        row = dict(raw)
        row['_local_date'] = dt(row['created_at']).astimezone(TZ).date()
        row['_usd'] = dec(row['amount']) * dec(rates[row['currency']])
        converted.append(row)

    ROWS = converted
    RATES = rates
    PRODUCTS = sorted({r['product'] for r in converted}, key=len, reverse=True)
    READY = True
    print(f'Loaded {len(ROWS)} current orders and {len(PRODUCTS)} products', flush=True)


@asynccontextmanager
async def lifespan(app):
    await load_data()
    yield


app = FastAPI(title='Acme Ledger Question Answering', lifespan=lifespan)


@app.get('/health')
async def health():
    return {'ready': READY, 'orders': len(ROWS)}


MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
REGIONS = ('north', 'south', 'east', 'west', 'central')


def parse_filters(question):
    lower = question.lower()
    region = next((r for r in REGIONS if re.search(r'\b' + r + r'\b', lower)), None)
    product = next((p for p in PRODUCTS if re.search(r'(?<!\w)' + re.escape(p.lower()) + r'(?!\w)', lower)), None)
    customer_match = re.search(r'\bC\d+\b', question, re.I)
    customer = customer_match.group().upper() if customer_match else None

    month = next((number for name, number in MONTHS.items() if re.search(r'\b' + re.escape(name) + r'\b', lower)), None)
    year_match = re.search(r'\b20\d{2}\b', lower)
    year = int(year_match.group()) if year_match else None
    qmatch = re.search(r'\bq([1-4])\b|\b(first|second|third|fourth) quarter\b', lower)
    quarter = None
    if qmatch:
        quarter = int(qmatch.group(1)) if qmatch.group(1) else ['first', 'second', 'third', 'fourth'].index(qmatch.group(2)) + 1

    status = None
    if re.search(r'\brefund(?:ed|s)?\b', lower):
        status = 'refunded'
    elif re.search(r'\b(?:void|voided|cancelled|canceled)\b', lower):
        status = 'void'
    elif re.search(r'\bpaid\b', lower):
        status = 'paid'
    return region, product, customer, month, year, quarter, status


def answer_question(question):
    lower = question.strip().lower()
    region, product, customer, month, year, quarter, status = parse_filters(question)

    is_revenue = bool(re.search(r'\b(?:revenue|sales|turnover|earned|income|gross|net)\b', lower))
    is_average = bool(re.search(r'\b(?:average|avg|mean|per\s+(?:order|transaction))\b', lower))
    asks_count = bool(re.search(r'\b(?:how many|number of|count of|total number|count)\b', lower))
    monetary = bool(re.search(r'\b(?:usd|dollars?|money|amount|value|worth|cost|price)\b', lower))
    units = bool(re.search(r'\b(?:units?|quantit(?:y|ies)|items?)\b', lower))
    explicitly_all = bool(re.search(r'\b(?:all statuses|regardless of status|all orders|every order)\b', lower))

    # A request for "how many dollars" is monetary, not a request to count orders.
    wants_count = asks_count and not (is_average or monetary or is_revenue)
    if status is None and not explicitly_all and not wants_count:
        status = 'paid'

    rows = [r for r in ROWS if status is None or r['status'].lower() == status]
    if region:
        rows = [r for r in rows if r['region'].lower() == region]
    if product:
        rows = [r for r in rows if r['product'].lower() == product.lower()]
    if customer:
        rows = [r for r in rows if r['customer'].upper() == customer]
    if year is not None:
        rows = [r for r in rows if r['_local_date'].year == year]
    if month is not None:
        rows = [r for r in rows if r['_local_date'].month == month]
    if quarter is not None:
        rows = [r for r in rows if (r['_local_date'].month - 1) // 3 + 1 == quarter]

    # Ranking questions: product/customer/region with highest or lowest value, units, or order count.
    ranking = bool(re.search(r'\b(?:top|best|highest|most|leading|popular|biggest|largest|least|lowest|bottom|smallest)\b', lower))
    grouped_by = None
    if ranking:
        if re.search(r'\bproducts?\b', lower) and not product:
            grouped_by = 'product'
        elif re.search(r'\bcustomers?\b', lower) and not customer:
            grouped_by = 'customer'
        elif re.search(r'\bregions?\b', lower) and not region:
            grouped_by = 'region'

    if grouped_by:
        by = defaultdict(list)
        for row in rows:
            by[row[grouped_by]].append(row)
        count_metric = bool(re.search(r'\b(?:orders?|transactions?)\b', lower)) and not (monetary or is_revenue)

        def score(group):
            if units:
                return sum((dec(r['qty']) for r in group), Decimal(0))
            if count_metric:
                return Decimal(len(group))
            return sum((r['_usd'] for r in group), Decimal(0))

        descending = not bool(re.search(r'\b(?:least|lowest|bottom|smallest)\b', lower))
        ranked = sorted(by, key=lambda k: (-score(by[k]) if descending else score(by[k]), k))
        nmatch = re.search(r'\b(?:top|bottom)\s+(\d+)\b', lower)
        n = int(nmatch.group(1)) if nmatch else 1
        return ranked[:n] if n > 1 else (ranked[0] if ranked else None)

    # MUST precede count handling: "On average, how many US dollars ...?"
    if is_average:
        if not rows:
            return 0
        if units and not monetary and not is_revenue:
            return money(sum((dec(r['qty']) for r in rows), Decimal(0)) / Decimal(len(rows)))
        return money(sum((r['_usd'] for r in rows), Decimal(0)) / Decimal(len(rows)))

    if wants_count:
        if re.search(r'\b(?:distinct|unique)\s+customers?\b|\bcustomers?\b', lower) and not customer:
            return len({r['customer'] for r in rows})
        if re.search(r'\b(?:distinct|unique)\s+products?\b', lower):
            return len({r['product'] for r in rows})
        if units:
            return sum(int(r['qty']) for r in rows)
        return len(rows)

    if units and not (monetary or is_revenue):
        return sum(int(r['qty']) for r in rows)

    return money(sum((r['_usd'] for r in rows), Decimal(0)))


@app.post('/')
async def solve(body: Question):
    if not READY:
        raise HTTPException(503, 'Ledger not loaded')
    try:
        return {'answer': answer_question(body.question)}
    except Exception as exc:
        raise HTTPException(422, f'Unable to answer question: {type(exc).__name__}') from exc


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
