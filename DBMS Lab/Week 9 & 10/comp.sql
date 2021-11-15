create database company;

\c company

create table department (dname VARCHAR(15), emp_count INT, PRIMARY KEY (dname));

create table employee (fname VARCHAR(15) NOT NULL, lname VARCHAR(15) NOT NULL, essn CHAR(10) NOT NULL, dname VARCHAR(15), PRIMARY KEY (essn), FOREIGN KEY (dname) REFERENCES department(dname));


create function inc_213() returns trigger as $$
begin
update department set emp_count = emp_count + 1
where department.dname = NEW.dname;
return NEW;
end;
$$ LANGUAGE plpgsql;

create trigger increment_213
after insert on employee
for each row
execute procedure inc_213();

create function dec_213() returns trigger as $$
begin
update department set emp_count = emp_count - 1
where department.dname = OLD.dname;
return OLD;
end;
$$ LANGUAGE plpgsql;

create trigger decrement_213
after delete on employee
for each row
execute procedure dec_213();


INSERT into department values ('Poison', 0);
INSERT into department values ('Serum', 0);

INSERT into employee values ('Henry', 'Jekyll', '1234567890', 'Poison');
INSERT into employee values ('Edward', 'Hyde', '0123456789', 'Poison');
INSERT into employee values ('Gabriel', 'Utterson', '8264910382', 'Serum');
