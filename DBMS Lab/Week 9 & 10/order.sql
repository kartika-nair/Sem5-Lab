create database ord;

\c ord

create table order_summary (numItems INT, totalPrice INT, ID VARCHAR(5), PRIMARY KEY (ID));

create table order_item (name VARCHAR(15), qty INT, unitPrice INT, ID VARCHAR(5), PRIMARY KEY (name), FOREIGN KEY (ID) REFERENCES order_summary(ID));


create function inc_213() returns trigger as $$
begin
update order_summary set totalPrice = totalPrice + (NEW.qty * NEW.unitPrice)
where order_summary.ID = NEW.ID;
update order_summary set numItems = numItems + 1
where order_summary.ID = NEW.ID;
return NEW;
end;
$$ LANGUAGE plpgsql;

create trigger increment_213
after insert on order_item
for each row
execute procedure inc_213();

create function dec_213() returns trigger as $$
begin
update order_summary set totalPrice = totalPrice - (OLD.qty * OLD.unitPrice)
where order_summary.ID = OLD.ID;
update order_summary set numItems = numItems - 1
where order_summary.ID = OLD.ID;
return OLD;
end;
$$ LANGUAGE plpgsql;

create trigger decrement_213
after delete on order_item
for each row
execute procedure dec_213();


INSERT into order_summary values (0, 0, '12345');
INSERT into order_summary values (0, 0, '21345');
INSERT into order_summary values (0, 0, '31245');

INSERT into order_item values ('Hyde', 1, 2, '12345');
INSERT into order_item values ('Jekyll', 2, 1, '12345');
INSERT into order_item values ('Enfield', 3, 4, '21345');
INSERT into order_item values ('Utterson', 5, 3, '31245');
