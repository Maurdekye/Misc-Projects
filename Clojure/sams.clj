(defn row [N C]
  (cond
    (<= C 0)
      []
    :else
      (conj
      (row N (dec C))(+ (rand-int 20)10))))


(defn make_grid [N C]
  (cond
    (<= C 0)
    []
    :else
      (conj
      (make_grid N (dec C))(row N N))))

(defn visual [C grid]
  (cond
    (= C (count grid))
      (println "\n")
    :else
      (do
        (println (nth grid C))
        (visual (inc C) grid))))

(defn rotation [row]
  (cons (last row) (butlast row)))

(defn rotate_grid [grid irow]
  (assoc grid irow (rotation (nth grid irow))))
;;(def test_grid (make_grid 5 5))
;;(visual 0 test_grid)
;;(visual 0 (rotate_grid grid 0))

(defn my_compare [row1 row2 C]
  (cond
    (= C (count row1))
      false
    (= (nth row1 C)(nth row2 C))
      true
    :else
      (my_compare row1 row2 (inc C))))

(defn whole_compare [grid irow1 irow2]
  (cond
    (= (count grid) irow2)
      [grid false nil]
    (= irow1 irow2)
      (whole_compare grid irow1 (inc irow2))
    (my_compare (nth grid irow1) (nth grid irow2) 0)
      [(rotate_grid grid irow2) true irow2]
    :else
       (whole_compare grid irow1 (inc irow2))))

(let [[a b c] (whole_compare [[1 2 3] [4 5 6] [7 8 9]] 0 0)]
  (whole_compare a 0 1))

(defn change_top [grid irow]
  (if (= (count grid) irow)
    [grid false -1]
    (let [[new_grid has_changed changed_row] (whole_compare grid irow 0)]
      (if-not has_changed
        (change_top grid (inc irow))
        [new_grid true changed_row]))))

(defn solve [grid ilast_row C]
  (let [[new_grid has_changed changed_row] (change_top grid 0)]
    (cond
      (not has_changed)
         grid
      (= C (count grid))
        (println "couldnt solve")
      :else
      (if (= ilast_row changed_row)
        (solve new_grid changed_row (inc C))
        (solve new_grid changed_row 1)))))

(def grid (make_grid 5 5))
(visual 0 grid)
(visual 0 (solve grid -1 0))

