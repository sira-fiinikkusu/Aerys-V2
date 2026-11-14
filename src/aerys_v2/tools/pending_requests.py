"""D49: list or withdraw her approval requests through the existing approval door."""
from __future__ import annotations


def build_pending_requests_tool(approvals):
    from langchain_core.tools import tool

    @tool
    def my_pending_requests(action: str, approval_id: int | None = None) -> str:
        """Look at what I'm waiting for permission to do, or take back one request.

        Use list to see my pending requests and their ids. Use withdraw with one approval_id
        when I asked by mistake, the person no longer wants it, or the situation changed.
        I can do this on any turn, using my judgement. Withdrawing is final for that request;
        I can ask again later if needed. I cannot approve or decline requests myself.
        action: list or withdraw. approval_id: the single request to withdraw.
        """
        if action not in ('list', 'withdraw'):
            return 'Only list or withdraw is available. I cannot approve or decline requests.'
        if action == 'list':
            try:
                rows = approvals.store.pending(approvals.clock())
                if not rows:
                    return 'No pending approval requests.'
                return '\n'.join(
                    f'#{r.id} | {r.kind} | {r.summary} | asked {r.requested_at.isoformat()} | '
                    f'waiting on {", ".join(r.approvers) if r.approvers is not None else "the owner"}'
                    for r in rows)
            except Exception:
                return 'I could not read pending approval requests. Do not assume there are none.'
        if approval_id is None:
            return 'Not withdrawn: provide one approval_id from list.'
        # The store atomically checks pending AND unexpired; a competing tap can win first.
        if approvals.withdraw(approval_id):
            return (f'Withdrawn request #{approval_id}. It cannot execute from this request. '
                    'The watchers will void its card and drop any waiting action. '
                    'This is final for this request; I can ask again later if needed.')
        status = approvals.status(approval_id)
        if status == 'pending':
            return (f'Not withdrawn: I could not take back request #{approval_id} just now; it is still '
                    'waiting on his tap. Tell him not to approve it, or try again in a moment.')
        return (f'Not withdrawn: request #{approval_id} has status {status}. '
                'Only pending, unexpired requests can be withdrawn; do not claim it was cancelled.')

    return my_pending_requests
