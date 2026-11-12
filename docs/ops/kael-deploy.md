# kael-deploy: self-serve rack deploys for non-controversial changes

The owner's standing rule (2026-07-25): Kael decides which fixes Aerys gets; anything controversial needs the
owner's approval first, everything else ships with a receipt after. `kael-deploy <sha>` (a script on the rack host,
not in this repo) is how a non-controversial change ships without a per-change go. It refuses on its own unless:

- the commit is on a pushed branch and is a fast-forward of what is live;
- no file in the diff is on controversial ground: database migrations, her soul/config, the hands and outside-text
  gates, approvals, reach rules, identity and resolver, memory and extraction, contacts and outward messaging,
  privacy and location guards, test bodies, home jobs, or anything about money;
- a receipt for that exact commit records the full suite green and a cross-review with no open blockers;
- she is not in the middle of a voice conversation.

After the deploy, every live check must pass and no error lines may appear; otherwise it rolls back to the previous
image and checkout by itself. Everything it refuses still goes to the owner as a per-change go line.

Every run, refused or not, leaves one line in `~/bench/kael-deploy.log` on the rack host: REFUSED (and why),
ROLLED BACK (and which check failed), or the live revision it moved to.
