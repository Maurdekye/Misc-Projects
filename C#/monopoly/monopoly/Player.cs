using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace monopoly
{
    abstract class Player : Proprietor<Property>
    {
        public Game PlayingGame;
        public int Money;
        public int BoardPosition;
        public string Name { get; private set; }

        public bool Disqualified { get; private set; } = false;
        public List<Card> KeptCards = new List<Card>();
        public bool InJail = false;
        public int JailEscapeAttempts = 0;
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
            if (amount < 0)
                Charge(-amount);
            else
                Money += amount;
        }

        public int Charge(int amount)
        {
            Tell("You are charged ${amount}.");
            if (Money >= amount)
            {
                Money -= amount;
                return amount;
            }
            else
            {
                int fullAmount = amount;
                amount -= Money;
                Money = 0;
                Tell($"You were unable to afford the full charge.");
                while(CalculateTotalValueOfAssets() > 0)
                {
                    Tell($"you have ${amount} remaining to pay off.");
                    AskForPropertyToDowngrade().Downgrade();
                    if (Money >= amount)
                    {
                        Money -= amount;
                        return fullAmount;
                    }
                    else
                    {
                        amount -= Money;
                        Money = 0;
                    }
                }
                Tell("You ran out of assets while trying to pay off a charge; game over!");
                Disqualified = true;
                return fullAmount - amount;
            }

        }

        public void SendToJail()
        {
            if (InJail)
                return;
            for (int i = 0; i < KeptCards.Count; i++)
            {
                if (KeptCards[i] is GetOutOfJailCard)
                {
                    if (AskToUseGetOutOfJailFreeCard())
                    {
                        PlayingGame.RetakeCard(KeptCards[i]);
                        KeptCards.RemoveAt(i);
                        return;
                    }
                }
            }
            InJail = true;
            JailEscapeAttempts = 0;
            BoardPosition = PlayingGame.GameBoard.FindIndex(s => s is JailSpace);
        }

        public void MakeMovement()
        {
            int Roll1 = 0, Roll2 = 0;
            int ConsecutiveRolls = 0;
            while (Roll1 == Roll2)
            {
                ConsecutiveRolls += 1;
                if (ConsecutiveRolls == 4)
                {
                    Tell("You rolled doubles too much, and are sent to jail.");
                    SendToJail();
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

        public void MoveTo(Predicate<BoardSpace> test)
        {
            int currentPos = BoardPosition;
            for (int i = 0; i < PlayingGame.GameBoard.Count; i++)
            {
                currentPos = (currentPos + 1) % PlayingGame.GameBoard.Count;
                if (test.Invoke(PlayingGame.GameBoard[BoardPosition]))
                {
                    MoveTo(currentPos);
                    break;
                }
            }
        }

        public void TakeTurn()
        {
            while (true)
            {
                TurnActionResponse turnResponse = AskForTurnAction();
                if (turnResponse is TurnActionEnd)
                    break;
                else if (turnResponse is TurnActionForfeit)
                {
                    Disqualified = true;
                    return;
                }
                else if (turnResponse is TurnActionUpgradeProperty)
                {
                    (turnResponse as TurnActionUpgradeProperty).PropertyToUpgrade.Upgrade();
                }
                else if (turnResponse is TurnActionDowngradeProperty)
                {
                    (turnResponse as TurnActionDowngradeProperty).PropertyToDowngrade.Downgrade();
                }
                else if (turnResponse is TurnActionBuyPropertyFromPlayer)
                {
                    Property toBuy = (turnResponse as TurnActionBuyPropertyFromPlayer).PropertyToPurchase;
                    TradeResponse initialOfferRequest = AskForBetterTradeOffer(new List<TradingTender> { toBuy }, new List<TradingTender>());
                    if (initialOfferRequest is TradeRefusal)
                        continue;
                    TradeWithPlayer(toBuy.Owner as Player, new List<TradingTender> { toBuy }, (initialOfferRequest as TradeAccept).Contents);
                }
                else if (turnResponse is TurnActionBuyCardFromPlayer)
                {
                    TurnActionBuyCardFromPlayer buyCardResponse = turnResponse as TurnActionBuyCardFromPlayer;
                    GetOutOfJailCard toBuy = buyCardResponse.CardToPurchase;
                    TradeResponse initialOfferRequest = AskForBetterTradeOffer(new List<TradingTender> { toBuy }, new List<TradingTender>());
                    if (initialOfferRequest is TradeRefusal)
                        continue;
                    TradeWithPlayer(buyCardResponse.PlayerToBuyFrom, new List<TradingTender> { toBuy }, (initialOfferRequest as TradeAccept).Contents);
                }
                else if (turnResponse is TurnActionSellCardToPlayer)
                {
                    throw new NotImplementedException();
                }
                else if (turnResponse is TurnActionSellPropertyToPlayer)
                {
                    throw new NotImplementedException();
                }
            }
            if (InJail)
            {
                if (JailEscapeAttempts < 3)
                {
                    JailReleaseMethod releaseMethod = AskForJailReleaseMethod();
                    if (releaseMethod == JailReleaseMethod.PayFee)
                    {
                        Tell("You pay the $50 fee to bail yourself out.");
                        Charge(50);
                        InJail = false;
                    }
                    else if (releaseMethod == JailReleaseMethod.Roll)
                    {
                        int roll1 = Dice.Roll();
                        int roll2 = Dice.Roll();
                        if (roll1 == roll2)
                        {
                            Tell($"You roll double {roll1}'s, you were able to escape scott free!");
                            InJail = false;
                        }
                        else
                        {
                            Tell($"You rolled a {roll1} and a {roll2}, you stay in jail for another turn.");
                            JailEscapeAttempts += 1;
                        }
                    }
                }
                else
                {
                    Tell("You have made too many escape attempts, you're forced to pay the $50 fee to bail yourself out.");
                    Charge(50);
                    InJail = false;
                }
            }
            else
            {
                MakeMovement();
            }
        }

        public int CalculateTotalValueOfAssets()
        {
            int total = Money;
            total += GetProperties().Select(p => (p.IsMortgaged ? p.MortgageValue : p.Price)).Sum();
            total += Util.FilterSubclass<Property, ColoredProperty>(GetProperties()).Select(p => p.Houses * p.HouseCost).Sum();
            return total;
        }

        public void TradeWithPlayer(Player Other, List<TradingTender> Request, List<TradingTender> Offer)
        {
            TradeResponse response = Other.AskForTrade(Request, Offer);
            if (response is TradeRefusal)
            {
                TradeResponse betterTrade = AskForBetterTradeOffer(Request, Offer);
                if (betterTrade is TradeRefusal)
                    return;
                else
                    TradeWithPlayer(Other, Request, (betterTrade as TradeAccept).Contents);
            }
            else
            {
                List<TradingTender> tradeContents = (response as TradeAccept).Contents;
                TransferTender(Other, this, tradeContents);
                TransferTender(this, Other, Offer);
            }
        }

        public static void TransferTender(Player from, Player to, List<TradingTender> tender)
        {
            foreach (TradingTender item in tender)
            {
                if (item is MoneyTradingTender)
                {
                    to.ModifyCapital(from.Charge((item as MoneyTradingTender).Amount));
                }
                else if (item is GetOutOfJailCard)
                {
                    from.KeptCards.Remove(item as GetOutOfJailCard);
                    to.KeptCards.Add(item as GetOutOfJailCard);
                }
                else if (item is Property)
                {
                    (item as Property).Owner = to;
                }
            }
        }

        public abstract void Tell(string message);

        public abstract bool AskToUseGetOutOfJailFreeCard();

        public abstract bool AskToBuyProperty(Property property);

        public abstract JailReleaseMethod AskForJailReleaseMethod();

        public abstract Property AskForPropertyToDowngrade();

        public abstract TurnActionResponse AskForTurnAction();

        public abstract AuctionResponse AskAuctionParticipation(Property auctioningProperty, int currentPrice);

        public abstract TradeResponse AskForTrade(List<TradingTender> Request, List<TradingTender> Offer);

        public abstract TradeResponse AskForBetterTradeOffer(List<TradingTender> Request, List<TradingTender> Offer);
    }

    enum JailReleaseMethod
    {
        Roll, PayFee
    }

    class AIPlayer : Player
    {

    }

    class HumanPlayer : Player
    {

    }

    abstract class AuctionResponse { }
    class AuctionPass : AuctionResponse
    {
        public AuctionPass() { }
    }
    class AuctionBuy : AuctionResponse
    {
        public int Amount { get; private set; }

        public AuctionBuy(int Amount)
        {
            this.Amount = Amount;
        }
    }

    abstract class TurnActionResponse { }
    class TurnActionEnd : TurnActionResponse
    {
        public TurnActionEnd() { }
    }
    class TurnActionUpgradeProperty : TurnActionResponse
    {
        public Property PropertyToUpgrade { get; private set; }

        public TurnActionUpgradeProperty(Property PropertyToUpgrade)
        {
            this.PropertyToUpgrade = PropertyToUpgrade;
        }
    }
    class TurnActionDowngradeProperty : TurnActionResponse
    {
        public Property PropertyToDowngrade { get; private set; }

        public TurnActionDowngradeProperty(Property PropertyToDowngrade)
        {
            this.PropertyToDowngrade = PropertyToDowngrade;
        }
    }
    class TurnActionBuyPropertyFromPlayer : TurnActionResponse
    {
        public Property PropertyToPurchase { get; private set; }

        public TurnActionBuyPropertyFromPlayer(Property PropertyToPurchase)
        {
            this.PropertyToPurchase = PropertyToPurchase;
        }
    }
    class TurnActionBuyCardFromPlayer : TurnActionResponse
    {
        public GetOutOfJailCard CardToPurchase { get; private set; }
        public Player PlayerToBuyFrom { get; private set; }

        public TurnActionBuyCardFromPlayer(GetOutOfJailCard CardToPurchase, Player PlayerToBuyFrom)
        {
            this.CardToPurchase = CardToPurchase;
            this.PlayerToBuyFrom = PlayerToBuyFrom;
        }
    }
    class TurnActionSellPropertyToPlayer : TurnActionResponse
    {
        public Property PropertyToSell { get; private set; }
        public Player PlayerToSellTo { get; private set; }

        public TurnActionSellPropertyToPlayer(Property PropertyToSell, Player PlayerToSellTo)
        {
            this.PropertyToSell = PropertyToSell;
            this.PlayerToSellTo = PlayerToSellTo;
        }
    }
    class TurnActionSellCardToPlayer : TurnActionResponse
    {
        public GetOutOfJailCard CardToSell { get; private set; }
        public Player PlayerToSellTo { get; private set; }

        public TurnActionSellCardToPlayer(GetOutOfJailCard CardToSell, Player PlayerToSellTo)
        {
            this.CardToSell = CardToSell;
            this.PlayerToSellTo = PlayerToSellTo;
        }
    }
    class TurnActionForfeit : TurnActionResponse
    {
        public TurnActionForfeit() { }
    }

    public interface TradingTender { }
    class MoneyTradingTender : TradingTender
    {
        public int Amount { get; private set; }

        public MoneyTradingTender(int Amount)
        {
            this.Amount = Amount;
        }
    }

    abstract class TradeResponse { }
    class TradeRefusal : TradeResponse
    {
        public TradeRefusal() { }
    }
    class TradeAccept : TradeResponse
    {
        public List<TradingTender> Contents;

        public TradeAccept(List<TradingTender> Recieval)
        {
            this.Contents = Recieval;
        }
    }
}
