use bincode::{Decode, Encode};
use std::{
    collections::HashMap,
    fmt::{self, Display},
    fs::File,
    io,
    ops::{Add, Neg, Sub},
    path,
};

pub trait Zero {
    fn zero() -> Self;
}

pub trait Time:
    Zero + Sized + Display + Sub<Output = Self> + Add<Output = Self> + PartialOrd + Copy + Encode
{
    fn approx_eq(&self, other: &Self) -> bool;
    fn max_val() -> Self;
    fn is_max_val(&self) -> bool;
}

pub trait SignalVal:
    Zero + Sized + Display + Sub<Output = Self> + Neg<Output = Self> + PartialOrd + Copy + Encode
{
    fn max_val() -> Self;
    fn min_val() -> Self;
    fn is_max_val(&self) -> bool;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
}

impl Zero for f64 {
    fn zero() -> Self {
        0.
    }
}

impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}

impl Time for f64 {
    fn approx_eq(&self, other: &Self) -> bool {
        (self - other).abs() < 1e-10
    }

    fn max_val() -> Self {
        f64::INFINITY
    }

    fn is_max_val(&self) -> bool {
        self.is_infinite() && self.is_sign_positive()
    }
}

impl Time for i32 {
    fn approx_eq(&self, other: &Self) -> bool {
        self == other
    }

    fn max_val() -> Self {
        i32::MAX
    }

    fn is_max_val(&self) -> bool {
        self == &i32::MAX
    }
}

impl SignalVal for f64 {
    fn max_val() -> Self {
        f64::INFINITY
    }

    fn min_val() -> Self {
        f64::NEG_INFINITY
    }

    fn is_max_val(&self) -> bool {
        self.is_infinite() && self.is_sign_positive()
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
}

impl SignalVal for i32 {
    fn max_val() -> Self {
        i32::MAX
    }

    fn min_val() -> Self {
        i32::MIN
    }

    fn is_max_val(&self) -> bool {
        self == &i32::MAX
    }

    fn max(self, other: Self) -> Self {
        Ord::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        Ord::min(self, other)
    }
}

#[derive(Encode, Decode, Clone)]
pub struct Interval<T: Time> {
    pub closed_lb: bool,
    pub closed_ub: bool,
    pub lb: T,
    pub ub: T,
}

impl<T: Time> Interval<T> {
    pub fn new(closed_lb: bool, lb: T, ub: T, closed_ub: bool) -> Self {
        Self {
            closed_lb,
            closed_ub,
            lb,
            ub,
        }
    }
}

impl<T: Time> Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.closed_lb, self.closed_ub) {
            (true, true) => write!(f, "[{},{}]", self.lb, self.ub),
            (true, false) => write!(f, "[{},{})", self.lb, self.ub),
            (false, true) => write!(f, "({},{}]", self.lb, self.ub),
            (false, false) => write!(f, "({},{})", self.lb, self.ub),
        }
    }
}

#[derive(Encode, Decode, Clone, Copy)]
pub enum Comparison {
    GT,
    LTE,
}

impl Comparison {
    fn new(cmp: &str) -> Result<Self, &'static str> {
        match cmp {
            ">" => Ok(Self::GT),
            "<=" => Ok(Self::LTE),
            _ => Err("Invalid comparison operator"),
        }
    }
}

impl Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GT => write!(f, ">"),
            Self::LTE => write!(f, "<="),
        }
    }
}

#[derive(Encode, Decode, Clone)]
pub struct Predicate<S: SignalVal> {
    pub id: String,
    pub cmp: Comparison,
    pub val: S,
}

impl<S: SignalVal> Predicate<S> {
    pub fn new(id: &str, cmp: &str, val: S) -> Self {
        Self {
            id: String::from(id),
            cmp: Comparison::new(cmp).expect("Predicate creation failure"),
            val,
        }
    }
}

impl<S: SignalVal> Display for Predicate<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x {} {}", self.cmp, self.val)
    }
}

#[derive(Encode, Decode)]
pub enum FormulaSymbol<T: Time> {
    True,
    Pred(String),
    Neg,
    And,
    Or,
    Until(Interval<T>),
    Future(Interval<T>),
    Global(Interval<T>),
}

impl<T: Time> Display for FormulaSymbol<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let unbounded = |ivl: &Interval<T>| {
            // Returns true if ivl is [0, inf), false otherwise
            ivl.closed_lb && T::approx_eq(&ivl.lb, &T::zero()) && ivl.ub.is_max_val()
        };
        match self {
            Self::True => write!(f, "true"),
            Self::Pred(id) => write!(f, "{}", id),
            Self::Neg => write!(f, "!"),
            Self::And => write!(f, r"/\"),
            Self::Or => write!(f, r"\/"),
            Self::Until(ivl) => {
                if unbounded(ivl) {
                    write!(f, "U")
                } else {
                    write!(f, "U_{}", ivl)
                }
            }
            Self::Future(ivl) => {
                if unbounded(ivl) {
                    write!(f, "<>")
                } else {
                    write!(f, "<>_{}", ivl)
                }
            }
            Self::Global(ivl) => {
                if unbounded(ivl) {
                    write!(f, "[]")
                } else {
                    write!(f, "[]_{}", ivl)
                }
            }
        }
    }
}

#[derive(Encode, Decode)]
pub struct Formula<S: SignalVal + 'static, T: Time + 'static> {
    pub symbols: Vec<Option<FormulaSymbol<T>>>,
    pub preds: HashMap<String, Predicate<S>>,
}

impl Formula<i32, i32> {
    pub fn sv_format(&self) -> String {
        /* Expected SystemVerilog values for `FormulaSymbol`s:
         * True: 0
         * Pred: 1
         * Neg: 2
         * And: 3
         * Or: 4
         * Until: 5
         * Future: 6
         * Global: 7
         */
        use FormulaSymbol as FS;
        let (formula_type, formula_val): (Vec<_>, Vec<_>) = self
            .symbols
            .iter()
            .map(|o| match o {
                Some(FS::True) | None => Ok((String::from("0"), String::from("0,0"))),
                Some(FS::Pred(id)) => match self.preds[id] {
                    Predicate {
                        id: _,
                        cmp: Comparison::GT,
                        val,
                    } => Ok((String::from("1"), format!("{},0", val))),
                    _ => Err("predicate uses <= comparison"),
                },
                Some(FS::Neg) => Ok((String::from("2"), String::from("0,0"))),
                Some(FS::And) => Ok((String::from("3"), String::from("0,0"))),
                Some(FS::Or) => Ok((String::from("4"), String::from("0,0"))),
                Some(FS::Until(ivl)) => Ok((String::from("5"), format!("{},{}", ivl.lb, ivl.ub))),
                Some(FS::Future(ivl)) => Ok((String::from("6"), format!("{},{}", ivl.lb, ivl.ub))),
                Some(FS::Global(ivl)) => Ok((String::from("7"), format!("{},{}", ivl.lb, ivl.ub))),
            })
            .collect::<Result<Vec<_>, &'static str>>()
            .expect("Cannot generate a SystemVerilog string from formula")
            .into_iter()
            .unzip();

        format!("{},{}", formula_type.join(","), formula_val.join(","))
    }
}

impl<S: SignalVal, T: Time> Formula<S, T> {
    pub fn new(symbols: Vec<Option<FormulaSymbol<T>>>, preds: Vec<Predicate<S>>) -> Self {
        let mut map = HashMap::new();
        for pred in preds {
            map.insert(pred.id.clone(), pred)
                .map(|pred| pred.id)
                .ok_or("unique")
                .expect_err("Duplicated predicate id");
        }
        Self {
            symbols,
            preds: map,
        }
    }

    pub fn tree_string(&self) -> String {
        self.subtree_string(0)
    }

    fn subtree_string(&self, i: usize) -> String {
        use FormulaSymbol as FS;

        match &self.symbols[i] {
            Some(x @ (FS::True | FS::Pred(_))) => x.to_string(),
            Some(x @ FS::Neg) => format!("{}({})", x, self.subtree_string(2 * i + 1)),
            Some(x @ (FS::And | FS::Or | FS::Until(_))) => format!(
                "({}) {} ({})",
                self.subtree_string(2 * i + 1),
                x,
                self.subtree_string(2 * i + 2)
            ),
            Some(x @ (FS::Future(_) | FS::Global(_))) => {
                format!("{} ({})", x, self.subtree_string(2 * i + 1))
            }
            None => String::from(""),
        }
    }

    pub fn is_valid(&self) -> bool {
        use FormulaSymbol as FS;

        let n = self.symbols.len();
        // if (n + 1) is not a power of 2
        if n != usize::MAX && (n & (n + 1)) != 0 {
            return false;
        }

        for (i, elem) in self.symbols.iter().enumerate() {
            if !match &elem {
                Some(FS::True) | None => {
                    /* No children */
                    self.symbols
                        .get(2 * i + 1)
                        .map(Option::as_ref)
                        .flatten()
                        .is_none()
                        && self
                            .symbols
                            .get(2 * i + 2)
                            .map(Option::as_ref)
                            .flatten()
                            .is_none()
                }
                Some(FS::Pred(id)) => {
                    /* No children */
                    self.preds.contains_key(id)
                        && self
                            .symbols
                            .get(2 * i + 1)
                            .map(Option::as_ref)
                            .flatten()
                            .is_none()
                        && self
                            .symbols
                            .get(2 * i + 2)
                            .map(Option::as_ref)
                            .flatten()
                            .is_none()
                }
                Some(FS::Neg) => {
                    /* Only a left child */
                    self.symbols
                        .get(2 * i + 1)
                        .map(Option::as_ref)
                        .flatten()
                        .is_some()
                        && self
                            .symbols
                            .get(2 * i + 2)
                            .map(Option::as_ref)
                            .flatten()
                            .is_none()
                }
                Some(FS::And | FS::Or) => {
                    /* Two children */
                    self.symbols
                        .get(2 * i + 1)
                        .map(Option::as_ref)
                        .flatten()
                        .is_some()
                        && self
                            .symbols
                            .get(2 * i + 2)
                            .map(Option::as_ref)
                            .flatten()
                            .is_some()
                }
                Some(FS::Until(ivl)) => {
                    /* Two children */
                    ivl.lb < ivl.ub
                        && ivl.lb >= T::zero()
                        && self
                            .symbols
                            .get(2 * i + 1)
                            .map(Option::as_ref)
                            .flatten()
                            .is_some()
                        && self
                            .symbols
                            .get(2 * i + 2)
                            .map(Option::as_ref)
                            .flatten()
                            .is_some()
                }
                Some(FS::Future(ivl) | FS::Global(ivl)) => {
                    /* Only a left child */
                    ivl.lb < ivl.ub
                        && ivl.lb >= T::zero()
                        && self
                            .symbols
                            .get(2 * i + 1)
                            .map(Option::as_ref)
                            .flatten()
                            .is_some()
                        && self
                            .symbols
                            .get(2 * i + 2)
                            .map(Option::as_ref)
                            .flatten()
                            .is_none()
                }
            } {
                return false;
            }
        }
        return true;
    }
}

pub fn export_to_file<P, S, T>(
    path: P,
    contents: Vec<Formula<S, T>>,
) -> Result<usize, bincode::error::EncodeError>
where
    P: AsRef<path::Path>,
    S: SignalVal,
    T: Time,
{
    let mut file =
        File::create(path).map_err(|e| bincode::error::EncodeError::Io { inner: e, index: 0 })?;
    let config = bincode::config::standard();
    bincode::encode_into_std_write(contents, &mut file, config)
}

pub fn import_from_file<P, S, T>(path: P) -> Result<Vec<Formula<S, T>>, bincode::error::DecodeError>
where
    P: AsRef<path::Path>,
    S: SignalVal + Decode,
    T: Time + Decode,
{
    let file = File::open(path).map_err(|e| bincode::error::DecodeError::Io {
        inner: e,
        additional: usize::MAX,
    })?;
    let buffer = io::BufReader::new(file);
    let config = bincode::config::standard();
    bincode::decode_from_reader(buffer, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid() {
        use FormulaSymbol as FS;

        let mut phi: Formula<_, f64> = Formula::new(
            vec![
                Some(FS::Or),
                Some(FS::Pred(String::from("a"))),
                Some(FS::Neg),
                None,
                None,
                Some(FS::Pred(String::from("b"))),
            ],
            vec![Predicate::new("b", ">", 5.)],
        );
        assert!(!phi.is_valid());

        phi.symbols.push(None);
        assert!(!phi.is_valid());

        phi.preds
            .insert(String::from("a"), Predicate::new("a", ">", 3.1));
        assert!(phi.is_valid());
    }

    #[test]
    fn tree_string() {
        use FormulaSymbol as FS;

        let phi: Formula<_, f64> = Formula::new(
            vec![
                Some(FS::Or),
                Some(FS::Pred(String::from("a"))),
                Some(FS::Neg),
                None,
                None,
                Some(FS::Pred(String::from("b"))),
                None,
            ],
            vec![Predicate::new("b", ">", 5.), Predicate::new("a", ">", 3.1)],
        );
        assert!(phi.is_valid());
        assert_eq!(phi.tree_string(), r"(a) \/ (!(b))");

        let phi = Formula::new(
            vec![
                Some(FS::Until(Interval::new(true, 0., 5., false))),
                Some(FS::True),
                Some(FS::Pred(String::from("b"))),
            ],
            vec![Predicate::new("b", ">", 5.)],
        );
        assert!(phi.is_valid());
        assert_eq!(phi.tree_string(), "(true) U_[0,5) (b)");
    }

    #[test]
    fn sv_format() {
        use FormulaSymbol as FS;

        let phi = Formula::new(
            vec![
                Some(FS::Until(Interval::new(true, 0, 5, false))),
                Some(FS::True),
                Some(FS::Neg),
                None,
                None,
                Some(FS::Pred(String::from("a"))),
                None,
            ],
            vec![Predicate::new("a", ">", 3)],
        );
        assert!(phi.is_valid());
        assert_eq!(phi.sv_format(), "4,0,2,0,0,1,0,0,5,0,0,0,0,0,0,0,0,3,0,0,0");
    }
}
