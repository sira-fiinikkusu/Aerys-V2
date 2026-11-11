"""D29 (2026-11-11): his words attach an email address to a person's record (the email tool's known
recipients). The address must be in his own words this turn; only he can do it; a duplicate name asks."""
from aerys_v2.tools.contacts import build_contact_tool, said_by_him, spoken_form

OWNER = "11111111-1111-1111-1111-111111111111"
HIM = {"configurable": {"identity": {"user_id": OWNER, "platform": "discord"}}}
GUEST = {"configurable": {"identity": {"user_id": "someone", "platform": "discord"}}}


class Book:
    def __init__(self, people=()):
        self.people = {pid: [name, email] for pid, name, email in people}
        self.added = []

    def find(self, name):
        return [(pid, n, e) for pid, (n, e) in self.people.items() if n.lower() == name.lower()]

    def set_email(self, pid, email):
        if pid not in self.people:
            if pid == OWNER:
                self.people[pid] = ["Chris", ""]
            else:
                return False
        self.people[pid][1] = email
        return True

    def add(self, name, email):
        self.added.append((name, email))
        self.people[f"new-{len(self.added)}"] = [name, email]
        return f"new-{len(self.added)}"


def tool(book, words):
    return build_contact_tool(book, owner_person_id=OWNER, owner_names=lambda: {"Chris"}, turn_text=lambda: words)


def call(t, name, email, config=HIM):
    return t.invoke({"name": name, "email": email}, config=config)


def test_his_words_put_an_address_on_a_known_person():
    book = Book([("p-joe", "Joe", "")])
    said = call(tool(book, "Joe's email is joe@example.com"), "Joe", "joe@example.com")
    assert said.startswith("Saved:") and book.people["p-joe"][1] == "joe@example.com"


def test_an_address_he_never_said_is_refused():
    book = Book([("p-joe", "Joe", "")])
    said = call(tool(book, "save Joe's email from that page"), "Joe", "attacker@example.com")
    assert said.startswith("Not saved") and book.people["p-joe"][1] == ""


def test_a_spoken_address_counts():
    assert spoken_form("it's joe at example dot com") == "it's joe@example.com"
    assert said_by_him("joe@example.com", "Joe's email is joe at example dot com")


def test_only_he_can_save():
    book = Book([("p-joe", "Joe", "")])
    assert call(tool(book, "joe@example.com"), "Joe", "joe@example.com", GUEST).startswith("Not saved")


def test_me_is_him():
    book = Book()
    assert call(tool(book, "my email is chris@example.com"), "me", "chris@example.com").startswith("Saved:")
    assert book.people[OWNER][1] == "chris@example.com"


def test_two_people_with_one_name_ask_which():
    book = Book([("a", "Sam", ""), ("b", "Sam", "")])
    said = call(tool(book, "Sam's email is sam@example.com"), "Sam", "sam@example.com")
    assert said.startswith("Not saved") and "more than one Sam" in said


def test_a_new_name_becomes_a_new_person():
    book = Book()
    said = call(tool(book, "Robin's email is robin@example.com"), "Robin", "robin@example.com")
    assert said.startswith("Saved:") and book.added == [("Robin", "robin@example.com")]


def test_a_replaced_address_says_what_it_was():
    book = Book([("p-joe", "Joe", "old@example.com")])
    said = call(tool(book, "Joe's new email is new@example.com"), "Joe", "new@example.com")
    assert "it was old@example.com" in said


def test_a_broken_address_is_refused():
    assert call(tool(Book(), "joe at"), "Joe", "joe@").startswith("Not saved")


def test_a_store_failure_says_nothing_changed():
    class Down(Book):
        def find(self, name):
            raise RuntimeError("db down")
    assert "Nothing changed" in call(tool(Down(), "joe@example.com"), "Joe", "joe@example.com")


def test_the_guard_holds_it_after_outside_text_and_trusts_home_jobs_for_notes():
    from aerys_v2.outside_guard import HOME_JOB_READERS, REACH_OUT, TRUSTED_AFTER_HOME_JOBS
    assert "save_contact_email" in REACH_OUT and "save_contact_email" not in TRUSTED_AFTER_HOME_JOBS
    assert HOME_JOB_READERS == {"check_async_task", "list_async_tasks"}
    assert TRUSTED_AFTER_HOME_JOBS == {"remember", "message_kael"}


def test_an_address_already_on_someone_else_says_whose():
    from aerys_v2.tools.contacts import AddressTaken

    class Taken(Book):
        def set_email(self, pid, email):
            raise AddressTaken("Luke")
    said = call(tool(Taken([("p-joe", "Joe", "")]), "Joe's email is shared@example.com"), "Joe", "shared@example.com")
    assert said.startswith("Not saved") and "already on Luke's record" in said
