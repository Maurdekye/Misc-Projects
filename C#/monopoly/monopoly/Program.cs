using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    class Program
    {
        static void Main(string[] args)
        {
        }
    }

    class Game
    {
        public List<Property> PropertyList;
        public List<BoardSpace> GameBoard;
        public List<Player> Players;
        public Deck CommunityChest;
        public Deck Chance;
        public Bank GameBank;

        public void LoadCommunityChestDeck()
        {
            CommunityChest = new Deck();
            CommunityChest.Enqueue(new Card(DeckType.CommunityChest, a =>
            {

            }));
        }

        public void RetakeCard(Card c)
        {
            if (c.FromDeck == DeckType.CommunityChest)
                CommunityChest.Enqueue(c);
            else if (c.FromDeck == DeckType.Chance)
                Chance.Enqueue(c);
        }
    }

    /*
     class AdvanceToGoCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Advance Directly to Go");
            actor.MoveTo<GoSpace>();
        }
    }
    class BankErrorInFavorCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Bank error in your favor - collect $200");
            actor.ModifyCapital(200);
        }
    }
    class DoctorFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Doctor's fee - pay $50");
            actor.Charge(50);
        }
    }
    class GetOutOfJailFreeCommunityChestCard : Card, KeeperCard, GetOutOfJailCard
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Get out of jail free; you can keep this card until you need it");
        }
    }
    class GoToJailCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Go directly to jail.");
            actor.MoveTo<JailSpace>();
        }
    }
    class OperaNightCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Opera night; collect $50 from everyone else");
            actor.ModifyCapital(actor.PlayingGame.Players.Select(p => p == actor ? 0 : p.Charge(50)).Sum());
        }
    }
    class HolidayCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Holiday fund matures - collect $100");
            actor.ModifyCapital(100);
        }
    }
    class IncomeTaxRefundCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Income tax refund; collect $20");
            actor.ModifyCapital(20);
        }
    }
    class BirthdayCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("It's your birthday; collect $10 from all other players");
            actor.ModifyCapital(actor.PlayingGame.Players.Select(p => p == actor ? 0 : p.Charge(10)).Sum());
        }
    }
    class LifeInsuranceCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Life Insurance matures - collect $100");
            actor.ModifyCapital(100);
        }
    }
    class HospitalFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Play hostpital fees of $100");
            actor.Charge(100);
        }
    }
    class SchoolFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Play school fees of $150");
            actor.Charge(150);
        }
    }
    class ConsultancyFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Recieve $25 consultancy fee");
            actor.ModifyCapital(25);
        }
    }
    class StreetRepairsCard : Card
    {
        public override void CardAction(Player actor)
        {
            int amount = ((List<ColoredProperty>)actor.GetProperties().Where(p => p is ColoredProperty)).Select(p => p.Houses > 4 ? 115 : p.Houses * 40).Sum();
            if (amount > 0)
            {
                actor.Tell($"You are assesed for street repairs, and require a payment of ${amount}");
                actor.Charge(amount);
            }
            else
            {
                actor.Tell("You are assesed for street repairs, but are not deemed applicable for payment. You pay nothing.");
            }
        }
    }
    class BeautyContestCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("You have won second place in a beauty contest; collect $10");
            actor.ModifyCapital(10);
        }
    }
    class InheritanceCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("You inherit $100");
            actor.ModifyCapital(100);
        }
    } class AdvanceToGoCard : Card
    {
        public AdvanceToGoCard(DeckType FromDeck)
        {
            this.FromDeck = FromDeck;
        }

        public override void CardAction(Player actor)
        {
            actor.Tell("Advance Directly to Go");
            actor.MoveTo<GoSpace>();
        }
    }
    class BankErrorInFavorCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Bank error in your favor - collect $200");
            actor.ModifyCapital(200);
        }
    }
    class DoctorFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Doctor's fee - pay $50");
            actor.Charge(50);
        }
    }
    class GetOutOfJailFreeCommunityChestCard : Card, KeeperCard, GetOutOfJailCard
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Get out of jail free; you can keep this card until you need it");
        }
    }
    class GoToJailCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Go directly to jail.");
            actor.MoveTo<JailSpace>();
        }
    }
    class OperaNightCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Opera night; collect $50 from everyone else");
            actor.ModifyCapital(actor.PlayingGame.Players.Select(p => p == actor ? 0 : p.Charge(50)).Sum());
        }
    }
    class HolidayCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Holiday fund matures - collect $100");
            actor.ModifyCapital(100);
        }
    }
    class IncomeTaxRefundCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Income tax refund; collect $20");
            actor.ModifyCapital(20);
        }
    }
    class BirthdayCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("It's your birthday; collect $10 from all other players");
            actor.ModifyCapital(actor.PlayingGame.Players.Select(p => p == actor ? 0 : p.Charge(10)).Sum());
        }
    }
    class LifeInsuranceCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Life Insurance matures - collect $100");
            actor.ModifyCapital(100);
        }
    }
    class HospitalFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Play hostpital fees of $100");
            actor.Charge(100);
        }
    }
    class SchoolFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Play school fees of $150");
            actor.Charge(150);
        }
    }
    class ConsultancyFeeCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("Recieve $25 consultancy fee");
            actor.ModifyCapital(25);
        }
    }
    class StreetRepairsCard : Card
    {
        public override void CardAction(Player actor)
        {
            int amount = ((List<ColoredProperty>)actor.GetProperties().Where(p => p is ColoredProperty)).Select(p => p.Houses > 4 ? 115 : p.Houses * 40).Sum();
            if (amount > 0)
            {
                actor.Tell($"You are assesed for street repairs, and require a payment of ${amount}");
                actor.Charge(amount);
            }
            else
            {
                actor.Tell("You are assesed for street repairs, but are not deemed applicable for payment. You pay nothing.");
            }
        }
    }
    class BeautyContestCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("You have won second place in a beauty contest; collect $10");
            actor.ModifyCapital(10);
        }
    }
    class InheritanceCard : Card
    {
        public override void CardAction(Player actor)
        {
            actor.Tell("You inherit $100");
            actor.ModifyCapital(100);
        }
    }
    */

    // Property Implementation

    abstract class Property
    {
        public Proprietor<Property> Owner;
        public Game PlayingGame;
        public string Name;

        protected int Price;
        protected int MortgageValue;
        public bool IsMortgaged { get; private set; } = false;

        public bool CanUpgrade()
        {
            if (Owner.GetCapital() < GetUpgradePrice())
                return false;
            if (IsMortgaged)
                return true;
            return CanUpgrade_Internal();
        }
        protected abstract bool CanUpgrade_Internal();

        public int GetUpgradePrice()
        {
            if (IsMortgaged)
                return MortgageValue + (MortgageValue / 10);
            return  GetUpgradePrice_Internal();
        }
        protected abstract int GetUpgradePrice_Internal();

        public void Upgrade()
        {
            if (!CanUpgrade())
            {
                throw new InvalidOperationException("Attempted to upgrade property when unable to");
            }
            Owner.ModifyCapital(-GetUpgradePrice());
            if (IsMortgaged)
            {
                IsMortgaged = false;
            }
            else
            {
                Upgrade_Internal();
            }
        }
        protected abstract void Upgrade_Internal();

        protected abstract bool IsUpgraded();

        public bool CanDowngrade()
        {
            return IsUpgraded() || !IsMortgaged;
        }

        public int GetDowngradeValue()
        {
            if (!IsUpgraded() && !IsMortgaged)
                return MortgageValue;
            return GetDowngradeValue_Internal();
        }
        protected abstract int GetDowngradeValue_Internal();

        public void Downgrade()
        {
            if (!CanDowngrade())
            {
                throw new InvalidOperationException("Attempted to downgrade property when unable to");
            }
            Owner.ModifyCapital(GetDowngradeValue());
            if (!IsUpgraded())
            {
                IsMortgaged = true;
            }
            else
            {
                Downgrade_Internal();
            }
        }
        protected abstract void Downgrade_Internal();

        public int GetRent(Player victim)
        {
            if (IsMortgaged)
            {
                return 0;
            }
            else
            {
                return GetRent_Internal(victim);
            }
        }

        protected abstract int GetRent_Internal(Player victim);
    }

    enum Color
    {
        Brown, Blue, Pink, Orange, Red, Yellow, Green, Navy
    }

    class ColoredProperty : Property
    {
        public Color PropertyColor { get; private set; }
        private List<int> RentValues;
        private int HouseCost;

        public int Houses { get; private set; } = 0;

        public ColoredProperty(string Name, int Price, int MortgageValue, Proprietor<Property> InitialOwner, Color PropertyColor, List<int> RentValues, int HouseCost)
        {
            this.Name = Name;
            this.Price = Price;
            this.MortgageValue = MortgageValue;
            this.Owner = InitialOwner;
            this.PropertyColor = PropertyColor;
            this.RentValues = RentValues;
            this.HouseCost = HouseCost;
        }

        protected override int GetRent_Internal(Player victim)
        {
            int baseRent = RentValues[Houses] * 2;
            if (((List<ColoredProperty>)victim.PlayingGame.PropertyList.Where(p => p is ColoredProperty)).Any(p => p.PropertyColor == PropertyColor && p.Owner != Owner))
            {
                baseRent /= 2;
            }
            return baseRent;
        }

        protected override bool CanUpgrade_Internal()
        {
            return Houses < 5;
        }

        protected override int GetUpgradePrice_Internal()
        {
            return HouseCost;
        }

        protected override void Upgrade_Internal()
        {
            Houses += 1;
        }

        protected override bool IsUpgraded()
        {
            return Houses > 0;
        }

        protected override int GetDowngradeValue_Internal()
        {
            return HouseCost / 2;
        }

        protected override void Downgrade_Internal()
        {
            Houses -= 1;
        }
    }

    class StationProperty : Property
    {
        public StationProperty(string Name, int Price, int MortgageValue, Proprietor<Property> InitialOwner)
        {
            this.Name = Name;
            this.Price = Price;
            this.MortgageValue = MortgageValue;
            this.Owner = InitialOwner;
        }

        protected override bool CanUpgrade_Internal()
        {
            return false;
        }

        protected override void Downgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to downgrade a station property beyond mortgaging it");
        }

        protected override int GetDowngradeValue_Internal()
        {
            return 0;
        }

        protected override int GetRent_Internal(Player victim)
        {
            return (2^(victim.PlayingGame.PropertyList.Where(p => p is StationProperty && p.Owner == Owner).Count() - 1))*25;
        }

        protected override int GetUpgradePrice_Internal()
        {
            return 0;
        }

        protected override bool IsUpgraded()
        {
            return false;
        }

        protected override void Upgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to upgrade a station property beyond mortgaging it");
        }
    }

    class UtilityProperty : Property
    {
        public UtilityProperty(string Name, int Price, int MortgageValue, Proprietor<Property> InitialOwner)
        {
            this.Name = Name;
            this.Price = Price;
            this.MortgageValue = MortgageValue;
            this.Owner = InitialOwner;
        }

        protected override bool CanUpgrade_Internal()
        {
            return false;
        }

        protected override void Downgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to downgrade a utility property beyond mortgaging it");
        }

        protected override int GetDowngradeValue_Internal()
        {
            return 0;
        }

        protected override int GetRent_Internal(Player victim)
        {
            int baseRent = Dice.Roll();
            if (victim.PlayingGame.PropertyList.Where(p => p is UtilityProperty && p.Owner == Owner).Count() == 2)
                return baseRent * 10;
            else
                return baseRent * 4;
        }

        protected override int GetUpgradePrice_Internal()
        {
            return 0;
        }

        protected override bool IsUpgraded()
        {
            return false;
        }

        protected override void Upgrade_Internal()
        {
            throw new InvalidOperationException("Player should never be able to upgrade a utility property beyond mortgaging it");
        }
    }

    class Bank
    {

    }

    interface Proprietor<T>
    {
        IEnumerable<T> GetProperties();
        int GetCapital();
        void ModifyCapital(int amount);
    }

    // Board Implementation

    abstract class BoardSpace
    {
        public Game PlayingGame;

        public abstract void Land(Player arrival);
    }

    class PropertySpace : BoardSpace, Proprietor<Property>
    {
        public IEnumerable<Property> GetProperties()
        {
            foreach (Property p in PlayingGame.PropertyList)
            {
                if (p.Owner == this)
                    yield return p;
            }
        }

        public int GetCapital()
        {
            throw new InvalidOperationException("Attempted to get capital from board space");
        }

        public void ModifyCapital(int amount)
        {
            throw new InvalidOperationException("Attempted to modify capital of board space");
        }

        public override void Land(Player arrival)
        {
            throw new NotImplementedException();
        }
    }

    class JailSpace : BoardSpace
    {
        public override void Land(Player arrival)
        {
            arrival.Tell("You landed on jail, but you're just visiting.");
        }
    }

    class GotoJailSpace : BoardSpace
    {
        public override void Land(Player arrival)
        {
            arrival.Tell("You landed on Go Directly To Jail! You were sent to jail.");
            arrival.SendToJail();
        }
    }

    abstract class CardSpace : BoardSpace
    {
        public abstract string DrawCard(Player drawer);
    }

    class CommunityChestSpace : CardSpace
    {

    }

    class ChanceSpace : CardSpace
    {

    }

    abstract class TaxSpace : BoardSpace
    {

    }

    class IncomeTaxSpace : TaxSpace
    {

    }

    class LuxuryTaxSpace : TaxSpace
    {

    }

    class FreeParkingSpace : BoardSpace
    {

    }

    class GoSpace : BoardSpace
    {
        public override void Land(Player arrival) { }
    }

    // Card implementation

    class Deck : Queue<Card>
    {
        public void Draw(Player drawer)
        {
            Card drawn = Dequeue();
            drawn.Act(drawer);
            if (drawn is GetOutOfJailCard)
                drawer.KeptCards.Add(drawn);
            else
                Enqueue(drawn);
        }
    }
    
    public enum DeckType
    {
        CommunityChest, Chance
    }

    interface CardActor
    {
        void CardAction(Player actor);
    }

    class Card
    {
        public DeckType FromDeck;
        public CardActor Action;

        public Card(DeckType FromDeck, CardActor Action)
        {
            this.FromDeck = FromDeck;
            this.Action = Action;
        }

        public void Act(Player actor)
        {
            Action.CardAction(actor);
        }
    }
    
    class GetOutOfJailCard : Card
    {
        public GetOutOfJailCard(DeckType FromDeck, CardActor Action) : base(FromDeck, Action) { }
    }
    
    // Player Implementation

    abstract class Player : Proprietor<Property>
    {
        public Game PlayingGame;
        public int Money;
        public int BoardPosition;
        public string Name { get; private set; }

        public bool Disqualified { get; private set; } = false;
        public List<Card> KeptCards = new List<Card>();
        public bool InJail = false;
        public IEnumerable<Property> GetProperties()
        {
            foreach (Property p in PlayingGame.PropertyList)
            {
                if (p.Owner == this)
                    yield return p;
            }
        }

        public int GetCapital()
        {
            return Money;
        }

        public void ModifyCapital(int amount)
        {
            Money += amount;
        }

        public int Charge(int amount)
        {
            Money -= amount;
            return amount;
        }

        public void SendToJail()
        {
            for (int i = 0; i < KeptCards.Count; i++)
            {
                if (KeptCards[i] is GetOutOfJailCard)
                {
                    if(AskToUseGetOutOfJailFreeCard())
                    {
                        PlayingGame.RetakeCard(KeptCards[i]);
                        KeptCards.RemoveAt(i);
                        return;
                    }
                }
            }
            InJail = true;
            MoveTo<JailSpace>();
        }

        public void DoMovement()
        {
            int Roll1 = 0, Roll2 = 0;
            int ConsecutiveRolls = 0;
            while (Roll1 == Roll2)
            {
                ConsecutiveRolls += 1;
                if (ConsecutiveRolls == 4)
                {
                    SendToJail();
                    Tell("You rolled doubles too much, you were sent to jail.");
                    return;
                }
                Roll1 = Dice.Roll();
                Roll2 = Dice.Roll();
                int moveAmount = Roll1 + Roll2;
                Move(moveAmount);
                PlayingGame.GameBoard[BoardPosition].Land(this);
            }
        }

        public void Move(int spaces)
        {
            for (int i = 0; i < spaces; i++)
            {
                BoardPosition = (BoardPosition + 1) % PlayingGame.GameBoard.Count;
                if (PlayingGame.GameBoard[BoardPosition] is GoSpace)
                {
                    Tell("You passed go; you collect $200!");
                    Money += 200;
                }
            }
        }

        public void MoveTo(int position)
        {
            Move((position - BoardPosition) % PlayingGame.GameBoard.Count);
        }

        public void MoveTo<T>() where T : BoardSpace
        {
            int currentPos = BoardPosition;
            for (int i = 0; i < PlayingGame.GameBoard.Count; i++)
            {
                currentPos = (currentPos + 1) % PlayingGame.GameBoard.Count;
                if (PlayingGame.GameBoard[currentPos] is T)
                {
                    MoveTo(currentPos);
                    break;
                }
            }
        }

        public abstract void Tell(string message);

        public abstract bool AskToUseGetOutOfJailFreeCard();
    }

    class AIPlayer : Player
    {

    }

    class HumanPlayer : Player
    {

    }

    static class Dice
    {
        public static Random RNG = new Random();

        public static int Roll()
        {
            return Roll(1);
        }

        public static int Roll(int amount)
        {
            int dieSum = 0;
            for (int i = 0; i <amount;i++)
            {
                dieSum += RNG.Next(6);
            }
            return dieSum;
        }
    }

}
