(defun even (x)
    (eq (mod x 2) 0))

(defun sum (l)
    (if (null l)
        0
        (+ (car l) (sum (cdr l)))))

(defun rep (s e)
    (if (zerop s)
        nil
        (cons e (rep (- s 1) e))))

(defun plist (l)
    (if (not (null l))
        (progn 
            (princ (car l))
            (plist (cdr l)))
        t))

(defun deep-sum (l)
    (let ((fst (car l)) (rem (cdr l)))
        (cond
            ((null l)
                0)
            ((numberp fst)
                (+ fst (deep-sum rem)))
            ((listp fst)
                (+ (deep-sum fst) (deep-sum rem)))
            (t
                (deep-sum rem)))))

(defun intersperse (e l)
    (cond
        ((< (length l) 1)
            l)
        (t
            (cons e (cons (car l) (intersperse e (cdr l)))))))

(defun _deep-print (l ind)
    (let ((fst (car l)) (rem (cdr l)))
        (cond
            ((null l)
                nil)
            ((listp fst)
                (progn
                    (_deep-print fst (+ ind 1))
                    (_deep-print rem ind)))
            ((or (numberp fst) (symbolp fst))
                (progn
                    (plist (rep ind #\space))
                    (princ fst)
                    (terpri)
                    (_deep-print rem ind)))
            (t
                (_deep-print rem ind)))))

(defun deep-print (l)
    (_deep-print l 0))

(defun mat-print (m)
    (if (null m)
        nil
        (progn
            (plist (intersperse #\space (car m)))
            (terpri)
            (mat-print (cdr m)))))

(defun _range (n track)
    (if (eq track n)
        nil
        (cons track (_range n (+ track 1)))))

(defun range (n)
    (_range n 0))

(defun fold (comb start l)
    (if (null l)
        start
        (fold comb (funcall comb start (car l)) (cdr l))))

(defun filt (fn l)
    (cond
        ((null l)
            nil)
        ((funcall fn (car l))
            (cons (car l) (filt fn (cdr l))))
        (t
            (filt fn (cdr l)))))

(defun my-map (fn l)
    (if (null l)
        nil
        (cons (funcall fn (car l)) (my-map fn (cdr l)))))

(defun without (ind l)
    (cond
        ((< ind 1)
            (cdr l))
        (t
            (cons (car l) (without (- ind 1) (cdr l))))))

(defun elem (e l)
    (not (eq nil (member e l))))

(defun gind (e l)
    (if (not (elem e l))
        nil
        (- (length l) (length (member e l)))))

(defun subl (sl el)
    (cond
        ((null sl)
            t)
        ((null el)
            nil)
        ((elem (car sl) el)
            (subl (cdr sl) (without (gind (car sl) el) el)))
        (t
            nil)))

(defun devoid (e l)
    (cond
        ((null l)
            nil)
        ((listp (car l))
            (cons (devoid e (car l)) (devoid e (cdr l))))
        ((eq e (car l))
            (devoid e (cdr l)))
        (t
            (cons (car l) (devoid e (cdr l))))))

(defun ins-where (e c l)
    (cond
        ((null l)
            nil)
        ((listp (car l))
            (cons (ins-where e c (car l)) (ins-where e c (cdr l))))
        ((eq c (car l))
            (cons e (cons (car l) (ins-where e c (cdr l)))))
        (t
            (cons (car l) (ins-where e c (cdr l))))))

(defun list-lens (l)
    (if (null l)
        nil
        (cons (length (car l)) (list-lens (cdr l)))))

(defun max-num (l)
    (cond
        ((null l)
            nil)
        ((eq (length l) 1)
            (car l))
        (t
            (let ((mn (max-num (cdr l))))
                (if (> mn (car l))
                    mn
                    (car l))))))

(defun flatten (l)
    (cond 
        ((null l)
            nil)
        ((listp (car l))
            (append (flatten (car l)) (flatten (cdr l))))
        (t
            (cons (car l) (flatten (cdr l))))))

(defun qsort (l)
    (if (null l)
        nil
        (let ((sm (filt (lambda (x) (< x (car l))) (cdr l))) 
          (lg (filt (lambda (x) (>= x (car l))) (cdr l))))
        (append (qsort sm) (cons (car l) (qsort lg))))))